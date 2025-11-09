import os
import sys
import numpy as np
import pandas as pd
import pickle
import warnings
from collections import Counter # For custom RF compatibility

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# --- CONFIGURATION ---
PIPELINE_FILENAME = "fertilizer_recommendation_pipeline.pkl"

# --- 1. Load the Saved Pipeline ---
if not os.path.exists(PIPELINE_FILENAME):
    print(f"❌ ERROR: Pipeline file not found at '{PIPELINE_FILENAME}'")
    print("Please run 'python train_and_save.py' first to generate the model.")
    sys.exit(1)

try:
    with open(PIPELINE_FILENAME, 'rb') as f:
        pipeline_components = pickle.load(f)
    
    loaded_model = pipeline_components['model']
    loaded_le = pipeline_components['label_encoder']
    loaded_scaler = pipeline_components['scaler']
    feature_columns = pipeline_components['feature_columns']
    numerical_cols = pipeline_components['numerical_cols']
    target_names = pipeline_components['target_names']

    print(f"✅ Pipeline loaded successfully. Model type: {type(loaded_model).__name__}")

except Exception as e:
    print(f"❌ ERROR loading or parsing pipeline components: {e}")
    sys.exit(1)


# --- 2. Function to Process New Data Point ---

def process_new_data(new_data_dict, numerical_cols, feature_columns, scaler):
    """
    Processes a single new data point (dictionary) into the format the model expects.
    This ensures features are scaled and one-hot encoded correctly.
    """
    new_df = pd.DataFrame([new_data_dict])
    
    # 1. One-Hot Encode Categorical Features
    new_data_ohe = pd.get_dummies(new_df)
    
    # 2. Reindex to match the training feature columns (fills missing OHE columns with 0)
    X_processed = new_data_ohe.reindex(columns=feature_columns, fill_value=0)
    
    # 3. Scale Numerical Features using the loaded scaler
    if numerical_cols:
        X_processed[numerical_cols] = scaler.transform(X_processed[numerical_cols])
        
    # 4. Convert to NumPy array
    return X_processed.values


# --- 3. Example Data Point for Prediction ---

NEW_OBSERVATION = {
    'District_Name': 'Pune', 
    'Soil_color': 'Black', 
    'Nitrogen': 100, 
    'Phosphorus': 50, 
    'Potassium': 100, 
    'pH': 6.5, 
    'Rainfall': 1200, 
    'Temperature': 20, 
    'Crop': 'Sugarcane'
}


# --- 4. Make Prediction ---
print("\n--- Making Prediction for New Observation ---")
print(f"Input: {NEW_OBSERVATION}")

try:
    # 1. Process the new data
    X_new = process_new_data(
        NEW_OBSERVATION, 
        numerical_cols, 
        feature_columns, 
        loaded_scaler
    )
    
    # 2. Predict using the loaded model
    prediction_index = loaded_model.predict(X_new)[0]
    
    # 3. Decode the index back to the class name
    prediction_name = loaded_le.inverse_transform([prediction_index])[0]
    
    # --- 5. Output Result ---
    print("\n-------------------------------------------")
    print(f"✅ Prediction Complete:")
    print(f"Optimal Fertilizer Recommended: {prediction_name}")
    print("-------------------------------------------")

except Exception as e:
    print(f"❌ Prediction failed due to an error: {e}")
    sys.exit(1)
