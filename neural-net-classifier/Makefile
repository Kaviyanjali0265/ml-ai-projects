.PHONY: install train mlflow serve test docker-build docker-run

install:
	pip install -r requirements.txt

train:
	python src/train.py

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v

docker-build:
	docker build -t loan-approval-api .

docker-run:
	docker run -p 8000:8000 loan-approval-api
