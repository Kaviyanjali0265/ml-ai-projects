.PHONY: install train serve mlflow docker-build docker-run test

install:
	pip install -r requirements.txt

train:
	python src/train.py

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

docker-build:
	docker build -t churn-prediction-api .

docker-run:
	docker run -p 8000:8000 churn-prediction-api

test:
	pytest tests/ -v
