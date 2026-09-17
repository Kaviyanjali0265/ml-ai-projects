serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v

mlflow:
	mlflow ui --port 5000

docker-build:
	docker build -t llm-incident-analyzer .

docker-run:
	docker run -p 8000:8000 --add-host=host.docker.internal:host-gateway \
		-e OLLAMA_BASE_URL=http://host.docker.internal:11434/v1 \
		llm-incident-analyzer
