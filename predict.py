import cv2
import os
import sys
import numpy as np
import pickle
import warnings

# --- 1. Import Feature Extraction Dependencies ---
try:
    from skimage.feature import hog, graycomatrix, graycoprops, local_binary_pattern
except ImportError:
    print("Error: scikit-image not found. Please install it: pip install scikit-image")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# --- 2. Import Custom Classifier Class (Sequential) ---
try:
    # We need the source of the MLP class to correctly unpickle the model object
    from MLP import Sequential, Dense, Dropout 
    print("✅ Successfully imported custom Sequential model class.")
except ImportError as e:
    print(f"--- 🛑 IMPORT ERROR ---")
    print(f"Error importing 'Sequential' from 'MLP.py': {e}")
    print("Ensure 'MLP.py' is in the project root and needed classes are defined.")
    sys.exit(1)


# --- 3. Feature Extraction Functions (MUST match train.py) ---
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
        print(f"Error: Could not read image {img_path}.")
        return None
    
    img = cv2.resize(img, (128, 128))

    hog_f = extract_hog(img)
    glcm_f = extract_glcm_features(img)
    lbp_f = extract_lbp_features(img)
    stat_f = extract_statistical_features(img)

    combined = np.hstack([hog_f, glcm_f, lbp_f, stat_f])
    return combined

# --- 4. Load the Saved Pipeline ---
PIPELINE_FILENAME = "mlp_pipeline.pkl" 

try:
    with open(PIPELINE_FILENAME, 'rb') as f:
        # The pickle requires the MLP classes (Sequential, Dense, Dropout) to be defined 
        # in the scope for successful loading.
        pipeline = pickle.load(f)
    
    loaded_model = pipeline['model']
    loaded_le = pipeline['label_encoder']
    loaded_scaler = pipeline['scaler']
    loaded_pca = pipeline['pca']
    class_names = pipeline['class_names']
    
    print(f"✅ Successfully loaded pipeline from '{PIPELINE_FILENAME}'")
    
except FileNotFoundError:
    print(f"❌ ERROR: Pipeline file not found at '{PIPELINE_FILENAME}'")
    print("Please run 'python train.py' first to create the pipeline file.")
    sys.exit(1)
except Exception as e:
    print(f"❌ An error occurred while loading the pipeline: {e}")
    sys.exit(1)


# --- 5. Get Image Path from Command Line ---
if len(sys.argv) != 2:
    print("\n--- Usage ---")
    print(f"python {sys.argv[0]} <path_to_your_image>")
    print("Example: python predict.py Testing_Split/Blast Disease/B_1.jpg")
    sys.exit(1)

image_path_to_predict = sys.argv[1]

if not os.path.exists(image_path_to_predict):
    print(f"❌ ERROR: Image file not found at '{image_path_to_predict}'")
    sys.exit(1)


# --- 6. Make Prediction ---
print(f"\nProcessing '{image_path_to_predict}'...")

try:
    # 1. Extract features
    features = extract_features_from_image(image_path_to_predict)
    
    if features is not None:
        # 2. Reshape to (1, n_features)
        features_2d = features.reshape(1, -1)
        
        # 3. Apply the loaded scaler
        scaled_features = loaded_scaler.transform(features_2d)
        
        # 4. Apply the loaded PCA
        pca_features = loaded_pca.transform(scaled_features)
        
        # 5. Predict using the loaded custom model
        prediction_index = loaded_model.predict(pca_features)[0]
        
        # 6. Decode the index back to the class name
        prediction_name = loaded_le.inverse_transform([prediction_index])[0]
        
        # --- 7. Show Result ---
        print("\n--- ✅ Prediction Complete ---")
        print(f"  Image: {os.path.basename(image_path_to_predict)}")
        print(f"  Predicted Class: {prediction_name}")
        
except Exception as e:
    print(f"\n❌ An error occurred during processing or prediction: {e}")
    print("Ensure the image is valid and accessible and the pipeline file is intact.")
