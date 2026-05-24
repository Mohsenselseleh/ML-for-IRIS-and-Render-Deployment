# 🌸 Iris Classifier — FastAPI on Render

A Random Forest classifier trained on the Iris dataset, served via FastAPI.

## Endpoints

| Method | Path       | Description              |
|--------|------------|--------------------------|
| GET    | `/`        | API info                 |
| GET    | `/health`  | Health check             |
| GET    | `/docs`    | Interactive Swagger UI   |
| POST   | `/predict` | Classify an iris flower  |

## Example Request

```bash
curl -X POST https://your-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

## Example Response

```json
{
  "species": "setosa",
  "prediction": 0,
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  }
}
```

## Deploy to Render

1. Push this folder to a GitHub repo
2. Go to [render.com](https://render.com) → **New → Web Service**
3. Connect your repo — Render auto-detects `render.yaml`
4. Click **Deploy** — done!

The model trains automatically on first startup and is cached to disk.

## Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# Visit http://localhost:8000/docs
```
