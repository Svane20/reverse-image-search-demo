# Run FastAPI application

1. Install the requirements

```bash
pip install -r requirements.txt
```

2. Run the application

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Run Docker compose

1. Run the following command:

```bash
docker-compose up --build -d
```