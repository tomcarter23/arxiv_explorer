from pymongo.collection import Collection
import json
from bson.json_util import dumps


def collection_to_json(collection: Collection, output_path: str):
    cursor = collection.find({})
    with open(output_path, 'w') as file:
        json.dump(json.loads(dumps(cursor)), file)
