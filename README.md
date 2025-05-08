# SSEPT Recommendation API

A FastAPI-based API for serving recommendations from the SSEPT (Sequential recommendation with Self-attention) model.

## Project Structure

- `app.py` - API endpoints and configuration
- `utilities.py` - Model definitions and helper functions
- `requirements.txt` - Dependencies

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Make sure to install the dependencies for the correct Python environment that you will use to run the application.

## Running the Application

To start the server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Running in dockerfile

```
docker build -t project37-app:local .
docker run --rm -p 8000:8000 -p 3000:3000 project37-app:local
```


## API Endpoints

- **GET/POST /test**: Test endpoint to verify CORS functionality
- **POST /predict**: Main prediction endpoint
  - Request body:
    ```json
    {
      "user_id": 42,
      "sequence": [10, 11, 23, 99],
      "top_k": 5
    }
    ```
  - Response:
    ```json
    {
      "top_items": [42, 101, 56, 78, 23]
    }
    ```

## API Documentation

Once the server is running, you can access the auto-generated API documentation at:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc 