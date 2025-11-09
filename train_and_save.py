import numpy as np
import pandas as pd
import sys
import os
import pickle
import warnings
import time
from collections import Counter # For baseline check

# --- SKLEARN IMPORTS (for Preprocessing and Metrics only) ---
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier # For baseline check

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Import *CORRECTED* Custom Classifiers ---
try:
    # These MUST be the _FIXED versions you renamed
    from RandomForest import RandomForestClassifier
    from MultiLayerPerceptron import SimpleMLP
    print("Custom classifiers imported successfully.")
except ImportError as e:
    print(f"❌ ERROR: Could not import custom classifiers.")
    print("Please ensure RandomForest_FIXED.py and MultiLayerPerceptron_FIXED.py")
    print("have been renamed to RandomForest.py and MultiLayerPerceptron.py")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: A problem occurred in the custom classifier files: {e}")
    sys.exit(1)


# --- CONFIGURATION ---
DATA_PATH = "Crop and fertilizer dataset.csv"
OUTPUT_FILENAME = "fertilizer_recommendation_pipeline.pkl"
TARGET_COL = 'Fertilizer'
RANDOM_SEED = 42

def run_pipeline():
    """Executes the ML pipeline based on Kaggle notebook's preprocessing."""
    
    # --- 2. Data Loading ---
    print("\n--- 2. Data Loading ---")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at {DATA_PATH}.")
        sys.exit(1)

    # --- 3. Preprocessing (Kaggle Notebook Style) ---
    print("\n--- 3. Preprocessing (Kaggle Notebook Style) ---")
    
    # Remove columns deemed unnecessary in the notebook
    columns_to_remove = ['District_Name', 'Link']
    df = df.drop(columns=columns_to_remove)
    
    # Handle potential null values (if any)
    df = df.dropna()
    
    # --- Ordinal Encoding (Mapping) ---
    # This is the key technique from the notebook
    
    # Get all unique values to ensure mappings are complete
    soil_colors = df['Soil_color'].unique()
    crops = df['Crop'].unique()
    
    soil_color_mapping = {val: i for i, val in enumerate(soil_colors)}
    crop_mapping = {val: i for i, val in enumerate(crops)}
    
    df['Soil_color'] = df['Soil_color'].map(soil_color_mapping)
    df['Crop'] = df['Crop'].map(crop_mapping)
    
    print("Ordinal encoding complete. Features are now all numeric.")
    
    # Target Encoding (Y)
    le = LabelEncoder()
    df[TARGET_COL] = le.fit_transform(df[TARGET_COL])
    y = df[TARGET_COL].values
    NUM_CLASSES = len(le.classes_)
    print(f"Target classes: {NUM_CLASSES}")
    
    # Define Features (X)
    X_df = df.drop(columns=[TARGET_COL])
    X_feature_names = X_df.columns.tolist()
    
    # Feature Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df)
    N_FEATURES = X.shape[1]
    
    print(f"Feature matrix shape: {X.shape}. Total features: {N_FEATURES}")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # --- 4. Baseline Accuracy Check (The 53% Problem) ---
    print("\n--- 4. Baseline Accuracy Check ---")
    # Calculate most frequent class baseline
    most_frequent_class = Counter(y_train).most_common(1)[0]
    baseline_acc = most_frequent_class[1] / len(y_train)
    print(f"Most frequent class: {le.inverse_transform([most_frequent_class[0]])[0]} (Index {most_frequent_class[0]})")
    print(f"Baseline (predicting most frequent): {baseline_acc:.4f} (This is the ~53% score)")
    
    # Verify with DummyClassifier
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    dummy_acc = dummy_clf.score(X_test, y_test)
    print(f"DummyClassifier Test Accuracy: {dummy_acc:.4f} (Confirms baseline)")
    
    # --- 5. Model Optimization (Manual Grid Search) ---
    print("\n--- 5. Model Optimization (Using CORRECTED Custom Models) ---")
    results = {}
    best_score = -1
    best_model = None
    best_model_name = ""
    total_start_time = time.time()
    
    # --- 5a. Custom Random Forest Grid ---
    # With the new simpler feature set, this may work better
    rf_params_grid = {
        'n_trees': [50, 100, 200],
        'max_depth': [10, 20, 30]
    }
    
    for n_trees in rf_params_grid['n_trees']:
        for max_depth in rf_params_grid['max_depth']:
            model_name = f'RF_T{n_trees}_D{max_depth}'
            print(f"  Testing {model_name}...")
            
            rf_model = RandomForestClassifier(n_trees=n_trees, max_depth=max_depth)
            try:
                rf_model.fit(X_train, y_train)
                y_pred = rf_model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                print(f"    -> Accuracy: {score:.4f}") # Should be > 53% now
                results[model_name] = score

                if score > best_score:
                    best_score = score
                    best_model = rf_model
                    best_model_name = model_name
            except Exception as e:
                print(f"    -> Training failed for {model_name}. Error: {e}")

    # --- 5b. Custom SimpleMLP Grid (Aggressive) ---
    mlp_params_grid = {
        'n_hidden': [32, 64, 128],
        'learning_rate': [0.01, 0.05],
        'n_iters': [5000, 10000] # High iterations for convergence
    }
    
    for n_hidden in mlp_params_grid['n_hidden']:
        for lr in mlp_params_grid['learning_rate']:
            for n_iters in mlp_params_grid['n_iters']:
                model_name = f'MLP_H{n_hidden}_LR{lr}_I{n_iters}'
                print(f"  Testing {model_name}...")
                
                mlp_model = SimpleMLP(
                    n_input=N_FEATURES, n_hidden=n_hidden, n_output=NUM_CLASSES, 
                    learning_rate=lr, n_iters=n_iters, verbose=False, activation='relu' 
                )
                try:
                    mlp_model.fit(X_train, y_train) 
                    y_pred = mlp_model.predict(X_test)
                    score = accuracy_score(y_test, y_pred)
                    print(f"    -> Accuracy: {score:.4f}") # Should be > 53% now
                    results[model_name] = score

                    if score > best_score:
                        best_score = score
                        best_model = mlp_model
                        best_model_name = model_name
                except Exception as e:
                    print(f"    -> Training failed for {model_name}. Error: {e}")
                
    if best_model is None:
        print("\n❌ Training failed for all critical models.")
        sys.exit(1)

    total_end_time = time.time()
    print(f"\nTotal optimization time: {total_end_time - total_start_time:.2f} seconds.")
    print(f"\n✅ Optimization Complete. Best Model: {best_model_name} with Accuracy: {best_score:.4f}")
    
    # --- 6. Saving the Model Pipeline ---
    print("\n--- 6. Saving Model Pipeline ---")
    pipeline_components = {
        'model': best_model,
        'label_encoder_target': le, # For decoding predictions
        'scaler': scaler, 
        'feature_columns': X_feature_names, # e.g., ['Nitrogen', ..., 'Soil_color', 'Crop']
        'soil_color_mapping': soil_color_mapping, # Need this for new data
        'crop_mapping': crop_mapping, # Need this for new data
        'target_names': list(le.classes_)
    }
    
    try:
        with open(OUTPUT_FILENAME, 'wb') as f:
            pickle.dump(pipeline_components, f)
        print(f"✅ Successfully saved pipeline components to '{OUTPUT_FILENAME}'")
    except Exception as e:
        print(f"❌ ERROR saving pipeline: {e}")

if __name__ == "__main__":
    run_pipeline()