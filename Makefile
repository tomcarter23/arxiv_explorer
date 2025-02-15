.PHONY: e2e_tests_process
e2e_tests_process:
	docker compose -f docker-compose.process.e2e.yaml --profile e2e up --build --exit-code-from e2e-tests-process

.PHONY: process_example
process_example:
	PROCESS_ARGS="-d /data/input_data/arxiv-metadata-oai-snapshot.json -n 100 -fo /data/output_data/faiss_index.faiss -do /data/output_data/collection.json --log DEBUG" docker compose -f docker-compose.process.yaml --profile process up --build --exit-code-from run-process

.PHONY: process_all
process_example:
	PROCESS_ARGS="-d /data/input_data/arxiv-metadata-oai-snapshot.json -fo /data/output_data/faiss_index.faiss -do /data/output_data/collection.json --log DEBUG" docker compose -f docker-compose.process.yaml --profile process up --build --exit-code-from run-process

.PHONY: download_data
download_data:
	python arxiv_explorer/download_kaggle_data.py -o ./input_data
