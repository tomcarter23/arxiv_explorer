import json
import numpy as np
import os
import faiss
from pymongo import MongoClient
from pymongo.collection import Collection
import logging
from tqdm import tqdm
import argparse
from sentence_transformers import SentenceTransformer

from arxiv_explorer.mongo_utils import collection_to_json

logger = logging.getLogger(__name__)

MONGODB_URL = os.getenv("MONGODB_URL")


class DataHandler:

    def __init__(
        self,
        dataset: str,
        embedding_model: SentenceTransformer,
        faiss_index: faiss.Index,
        mongo_collection: Collection,
        num_to_proces: int = -1,
    ):
        self.dataset_path = dataset
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self.num_to_proces = num_to_proces
        self.mongo_collection = mongo_collection

    @staticmethod
    def load_data(dataset: str, n: int = -1):

        if n == 0:
            raise ValueError("n must be greater than 0, or -1 to process all documents")

        with open(dataset) as f:
            for i, line in enumerate(f):
                if i >= n > 0:
                    break
                yield json.loads(line)

    def get_embedding_vector(self, data: str) -> np.ndarray:
        embedding = self.embedding_model.encode(data)
        return np.array([embedding])

    def process_one(self, input_data: dict, i: int, attribute_to_encode: str = "abstract") -> None:
        embedding = self.get_embedding_vector(input_data[attribute_to_encode])
        self.faiss_index.add_with_ids(embedding, np.array([i]))
        input_data["faiss_id"] = i
        self.mongo_collection.insert_one(input_data)

    def process_and_save(self, index_path: str, attribute_to_encode: str = "abstract") -> None:
        logger.info("Beginning pipeline")

        for i, input_data in enumerate(tqdm(self.load_data(self.dataset_path, n=self.num_to_proces))):
            self.process_one(input_data, i, attribute_to_encode)

        faiss.write_index(self.faiss_index, index_path)
        logger.info(f"Done")


def setup_logging(level: str):
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s\t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)


def process_and_save(dataset_path: str, embedding_model: SentenceTransformer, faiss_index: faiss.Index, mongo_collection: Collection, output_path: str, num_to_proces: int = -1) -> None:
    dh = DataHandler(
        dataset=dataset_path,
        embedding_model=embedding_model,
        faiss_index=faiss_index,
        mongo_collection=mongo_collection,
        num_to_proces=num_to_proces,
    )
    dh.process_and_save(index_path=output_path)


def main() -> None:
    """Runs the script."""
    parser = argparse.ArgumentParser(
        description="Run processing pipeline to save embeddings and mongodb."
    )
    parser.add_argument(
        "--log",
        default="WARNING",
        help=f"Set the logging level. Available options: {list(logging._nameToLevel.keys())}",
    )
    parser.add_argument(
        "-n",
        default="-1",
        help=f"Number of documents to process. Default is -1, which processes all documents.",
    )
    parser.add_argument(
        "--transformer",
        "-t",
        default="all-MiniLM-L6-v2",
        help=f"Huggingface SentenceTransformer to use to calculate embeddings.",
    )
    parser.add_argument(
        "--mongo-db-col",
        "-mdc",
        default="arxivdb:arxivcol",
        help=f"String defining MongoDB database and collection to "
             f"save documents to. In the format <database>:<collection>.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        help=f"Path to Arxiv abstracts dataset to process."
    )
    parser.add_argument(
        "--faiss-output",
        "-fo",
        help=f"Path to save Faiss index to."
    )
    parser.add_argument(
        "--database-output",
        "-do",
        help=f"Path to save Mongo Collection index to JSON."
    )

    args = parser.parse_args()

    setup_logging(level=args.log)

    embedding_model = SentenceTransformer(args.transformer)
    faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension()))
    mongo_client = MongoClient(MONGODB_URL)
    database, collection = args.mongo_db_col.split(":")
    mongo_collection = mongo_client[database][collection]
    dataset_path = args.dataset

    process_and_save(
        dataset_path=dataset_path,
        embedding_model=embedding_model,
        faiss_index=faiss_index,
        mongo_collection=mongo_collection,
        output_path=args.faiss_output,
        num_to_proces=int(args.n),
    )

    if args.database_output:
        collection_to_json(mongo_collection, args.database_output)


if __name__ == "__main__":
    main()
