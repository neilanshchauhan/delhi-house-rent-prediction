from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- CORRECTED IMPORTS ---
from .inference import predict_rent, batch_predict_rent
from .schemas import RentPredictionRequest, PredictionResponse

# Initialize FastAPI app
app = FastAPI(
    title="Delhi House Rent Prediction API",
    description=(
        "An API to predict the monthly rent of residential properties in Delhi, India. "
        "This is a personal MLOps project to demonstrate model serving."
    ),
    version="1.0.0",
    contact={
        "name": "Neilansh Chauhan",
        "url": "https://github.com/neilansh/delhi-house-rent-prediction",
        "email": "neilanshchauhan4@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---
@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: RentPredictionRequest):
    return predict_rent(request)

@app.post("/batch-predict", response_model=list[PredictionResponse], tags=["Prediction"])
async def batch_predict_endpoint(requests: list[RentPredictionRequest]):
    return batch_predict_rent(requests)