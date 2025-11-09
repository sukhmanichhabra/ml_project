import cv2
import os
import sys
import numpy as np
import warnings
import pickle  
import time
import shutil
from tqdm import tqdm 
          #
# --- Import Sklearn Utilities for Preprocessing & Splitting (Not the Classifier) ---
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split 

# --- Import Feature Extraction Dependencies ---
try:
    from skimage.feature import hog, graycomatrix, graycoprops, local_binary_pattern
except ImportError:
    print("Error: scikit-image not found. Please install it: pip install scikit-image")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)


# --- 1. Import Your From-Scratch MLP (MLP.py must be defined separately) ---
try:
    # Assuming MLP.py is in the current working directory as per your structure
    from MLP import Sequential, Dense, Dropout 
    print("✅ Successfully imported custom MLP classifier (MLP.py).")
except ImportError as e:
    print(f"--- 🛑 IMPORT ERROR ---")
    print(f"Error importing 'Sequential, Dense, Dropout' from 'MLP.py': {e}")
    print("Please ensure 'MLP.py' is in the root project folder.")
    sys.exit(1)


# --- 2. Data Splitting Function (No Change) ---

def create_train_test_folders(source_dir, train_dir, test_dir, split=0.7):
    """Splits images from source_dir into Training and Testing folders."""
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print(f"Using existing '{train_dir}' and '{test_dir}' folders.")
        return

    print(f"\n--- Creating {split*100}% Train / { (1-split)*100}% Test Split ---")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    total_files = 0
    
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path): continue
        
        files = [f for f in os.listdir(class_path) if f.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'))]
        
        train_files, test_files = train_test_split(
            files, 
            train_size=split, 
            shuffle=True, 
            random_state=42
        )
        
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        for f in tqdm(train_files, desc=f'Copying {class_name} to Training'):
            shutil.copy(os.path.join(class_path, f), os.path.join(train_dir, class_name, f))
        
        for f in tqdm(test_files, desc=f'Copying {class_name} to Testing'):
            shutil.copy(os.path.join(class_path, f), os.path.join(test_dir, class_name, f))

        total_files += len(files)
        
    print(f"Split complete. Total files processed: {total_files}")


# --- 3. Feature Extraction Functions (omitted for brevity, assume consistency) ---
def extract_hog(img):
    hog_features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), block_norm='L2-Hys',
                          visualize=True, transform_sqrt=True)
    return hog_features

def extract_glcm_features(img):
    glcm = graycomatrix(img, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0]
    ]
    return features

def extract_lbp_features(img):
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, 11), 
                             range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6) 
    return hist

def extract_statistical_features(img):
    mean = np.mean(img)
    var = np.var(img)
    skew = np.mean((img - mean)**3) / (np.std(img)**3 + 1e-6)
    kurt = np.mean((img - mean)**4) / (np.std(img)**4 + 1e-6)
    
    hist = np.histogram(img.ravel(), 256, [0,256])[0]
    hist_norm = hist.astype("float") / (hist.sum() + 1e-6)
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-6))
    
    return [mean, var, skew, kurt, entropy]

def extract_features_from_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    img = cv2.resize(img, (128, 128))

    hog_f = extract_hog(img)
    glcm_f = extract_glcm_features(img)
    lbp_f = extract_lbp_features(img)
    stat_f = extract_statistical_features(img)

    combined = np.hstack([hog_f, glcm_f, lbp_f, stat_f])
    return combined

def load_dataset(folder_path):
    features, labels = [], []
    
    for label in os.listdir(folder_path):
        subfolder = os.path.join(folder_path, label)
        if not os.path.isdir(subfolder):
            continue
        
        print(f"  Processing class: {label}")
        for file in tqdm(os.listdir(subfolder), desc=f'  {label}'):
            img_path = os.path.join(subfolder, file)
            try:
                feature_vector = extract_features_from_image(img_path)
                if feature_vector is not None:
                    features.append(feature_vector)
                    labels.append(label)
            except Exception as e:
                continue
                
    return np.array(features), np.array(labels)

# --- 4. Execution ---

if __name__ == "__main__":
    
    SOURCE_DATA_PATH = "Rice_Diseases"
    TRAIN_DATA_PATH = "Training_Split" 
    TEST_DATA_PATH = "Testing_Split"

    if not os.path.isdir(SOURCE_DATA_PATH):
        print(f"Error: Source data folder '{SOURCE_DATA_PATH}' not found.")
        sys.exit(1)

    # 4a. Split the data
    create_train_test_folders(SOURCE_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH)
    
    # 4b. Load and extract features
    print(f"\nLoading and extracting features from '{TRAIN_DATA_PATH}'...")
    X_train, y_train_labels = load_dataset(TRAIN_DATA_PATH)
    
    print(f"\nLoading and extracting features from '{TEST_DATA_PATH}'...")
    X_test, y_test_labels = load_dataset(TEST_DATA_PATH)

    # --- Preprocessing Pipeline ---

    print("\nEncoding string labels to numbers...")
    le = LabelEncoder()
    y_train_num = le.fit_transform(y_train_labels)
    y_test_num = le.transform(y_test_labels)
    class_names = list(le.classes_)
    N_CLASSES = len(class_names)

    print("Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Using 120 components (safe limit)
    N_COMPONENTS = 120 
    print(f"Applying PCA (n_components={N_COMPONENTS})...")
    pca = PCA(n_components=N_COMPONENTS)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # --- ONE-HOT ENCODING (MANDATORY for Softmax Output) ---
    Y_train_one_hot = np.eye(N_CLASSES)[y_train_num]
    
    # --- 5. Define and Train Custom MLP Classifier (FINAL OPTIMAL TUNING) ---
    
    # FINAL TUNING
    N_INPUT = N_COMPONENTS
    
    # Optimized Learning Schedule
    LR = 0.002           
    EPOCHS = 3000         
    BATCH_SIZE = 128    
    L2_LAMBDA = 0.001
    CLIP_VALUE = 1.0

    print(f"\n--- Training Custom MLP Model (Layers: 256-128-64-64, Epochs={EPOCHS}, LR={LR}, PCA={N_COMPONENTS}, L2={L2_LAMBDA}) ---")
    
    # ARCHITECTURE
    model = Sequential(input_shape=N_INPUT) 

    # Layer 1
    model.add(Dense(N_INPUT, 256, activation='relu', lambda_param=L2_LAMBDA))
    model.add(Dropout(rate=0.4)) 

    # Layer 2
    model.add(Dense(256, 128, activation='relu', lambda_param=L2_LAMBDA))
    model.add(Dropout(rate=0.4)) 

    # Layer 3
    model.add(Dense(128, 64, activation='relu', lambda_param=L2_LAMBDA))
    model.add(Dropout(rate=0.4)) 
    
    # Layer 4 (WIDER final hidden layer)
    model.add(Dense(64, 64, activation='relu', lambda_param=L2_LAMBDA))
    
    # Output Layer
    model.add(Dense(64, N_CLASSES, activation='softmax', lambda_param=L2_LAMBDA))

    start_time = time.time()
    # Train model
    model.fit(
        X_train_pca, 
        Y_train_one_hot, 
        learning_rate=LR, 
        n_iters=EPOCHS, 
        batch_size=BATCH_SIZE,
        lambda_param=L2_LAMBDA,
        clip_value=CLIP_VALUE
    ) 
    duration = time.time() - start_time
    
    print("\n--- Predicting on Test Set ---")
    y_pred_num = model.predict(X_test_pca)

    # --- 6. Evaluate ---
    accuracy = accuracy_score(y_test_num, y_pred_num)
    print("\n--- Evaluation Complete ---")
    print(f"Model: Custom MLP (Layers: 256-128-64-64, Epochs={EPOCHS}, LR={LR}, PCA={N_COMPONENTS}, L2={L2_LAMBDA})")
    print(f"Training Time: {duration:.2f}s")
    print(f"Accuracy: {accuracy:.4f}") 

    print("\nClassification Report:\n")
    print(classification_report(y_test_num, y_pred_num, target_names=class_names))

    
    # --- 7. Save the Full Pipeline ---
    
    print("\n--- Saving Model & Pipeline ---")
    pipeline_components = {
        'model': model,
        'label_encoder': le,
        'scaler': scaler,
        'pca': pca,
        'class_names': class_names
    }
    
    output_filename = "mlp_pipeline.pkl"
    
    try:
        with open(output_filename, 'wb') as f:
            pickle.dump(pipeline_components, f)
        print(f"✅ Successfully saved pipeline components to '{output_filename}'")
    except Exception as e:
        print(f"--- 🛑 ERROR SAVING MODEL ---")
        print(f"An error occurred: {e}")
