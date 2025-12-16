import numpy as np
import pandas as pd
import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
MODEL_BASE_DIR = 'store_models'
TIME_STEP = 20
NUM_FEATURES = 4  # ['Weekly_Sales', 'Holiday_Flag', 'Sales_Lag_1', 'Sales_Rolling_Mean_4']

# Global Cache for Models & Scalers (Lazy Loading)
# Format: { store_id: {'model': model_obj, 'scaler': scaler_obj} }
model_cache = {}

# --- INPUT SCHEMA ---
class ForecastRequest(BaseModel):
    store_id: int
    # The input sequence must be a list of lists: 20 rows x 4 features
    # Example: [[sales_t-20, holiday, ...], ..., [sales_t-1, holiday, ...]]
    input_sequence: list[list[float]] 

# --- APP INITIALIZATION ---
app = FastAPI(title="Walmart AI Replenishment API")

# --- HELPER: MODEL LOADER ---
def get_store_resources(store_id: int):
    """
    Retrieves model/scaler from cache. If not present, loads from disk.
    """
    if store_id in model_cache:
        return model_cache[store_id]
    
    # Define Paths
    store_dir = os.path.join(MODEL_BASE_DIR, f'store_{store_id}')
    model_path = os.path.join(store_dir, f'lstm_model_store_{store_id}.h5')
    scaler_path = os.path.join(store_dir, f'scaler_store_{store_id}.pkl')

    # Check existence
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Artifacts for Store {store_id} not found.")

    print(f"📥 Loading resources for Store {store_id} into memory...")
    try:
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        # Cache resources
        model_cache[store_id] = {'model': model, 'scaler': scaler}
        return model_cache[store_id]
    except Exception as e:
        raise RuntimeError(f"Failed to load Store {store_id}: {e}")

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "active", "message": "Walmart Per-Store Forecasting API Ready"}

@app.post("/forecast")
def predict_store_sales(request: ForecastRequest):
    """
    Accepts 20 weeks of feature data for a specific store and returns the T+1 Forecast.
    """
    store_id = request.store_id
    raw_input = request.input_sequence

    # 1. Validation
    if len(raw_input) != TIME_STEP:
        raise HTTPException(status_code=400, detail=f"Input must have exactly {TIME_STEP} time steps.")
    
    try:
        # 2. Load Resources (Model + Scaler)
        resources = get_store_resources(store_id)
        model = resources['model']
        scaler = resources['scaler']
        
        # 3. Preprocessing
        # Convert list to numpy array (20, 4)
        input_array = np.array(raw_input)
        
        # Scale the input data
        # Note: The scaler expects (n_samples, 4). We pass our 20 rows.
        scaled_input = scaler.transform(input_array)
        
        # Reshape for LSTM: (1, 20, 4)
        lstm_input = scaled_input.reshape(1, TIME_STEP, NUM_FEATURES)
        
        # 4. Prediction
        scaled_prediction = model.predict(lstm_input, verbose=0)
        
        # 5. Inverse Transform
        # We need a dummy array (1, 4) to inverse transform the single prediction value
        dummy_pred = np.zeros((1, NUM_FEATURES))
        dummy_pred[:, 0] = scaled_prediction[0, 0] # Put prediction in the 1st column (Sales)
        
        final_forecast = scaler.inverse_transform(dummy_pred)[0, 0]
        
        return {
            "store_id": store_id,
            "status": "success",
            "weekly_sales_forecast": round(float(final_forecast), 2)
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model for Store {store_id} unavailable.")
    except Exception as e:
        print(f"Error processing Store {store_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))