# src/api/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# These imports will be created in the next steps
from .inference import predict_rent, batch_predict_rent
from .schemas import RentPredictionRequest, PredictionResponse

# Initialize the FastAPI app with metadata for documentation
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

# --- Middleware ---
# Add CORS (Cross-Origin Resource Sharing) middleware to allow requests
# from any origin. This is useful for development and for allowing
# a web client (like a Streamlit app) to communicate with the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- API Endpoints ---

@app.get("/health", tags=["Health Check"])
async def health_check():
    """
    Health check endpoint to confirm the API is running.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: RentPredictionRequest):
    """
    Endpoint for making a single rent prediction.
    Takes a JSON request with house features and returns a prediction.
    """
    # The core logic is handled by the `predict_rent` function in inference.py
    return predict_rent(request)


@app.post("/batch-predict", response_model=list[PredictionResponse], tags=["Prediction"])
async def batch_predict_endpoint(requests: list[RentPredictionRequest]):
    """
    Endpoint for making multiple rent predictions in a single batch request.
    Takes a list of JSON objects and returns a list of predictions.
    """
    # The core logic is handled by the `batch_predict_rent` function in inference.py
    return batch_predict_rent(requests)