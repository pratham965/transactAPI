import fastapi
import uvicorn

app = fastapi.FastAPI(title="TransactAI")

@app.get('/predict')
def predict():
    return {"server":"running"}

