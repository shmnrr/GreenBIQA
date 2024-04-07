import sys
import os

# Add the path to the ML_Model_CI directory
ml_model_ci_path = "/mnt/d/GitHub/ML_Model_CI"
sys.path.append(ml_model_ci_path)

from modelci.hub.converter import PyTorchConverter
import numpy as np
from ML_Model_CI.modelci.hub.converter import PyTorchConverter
import xgboost as xgb
from model import HybridFeatures
from core.utils.biqa import load_data, augment, jpeg_dct

# Load a single image from the data/test directory
data_dir = 'data/test'
images, mos = load_data(data_dir, load_mos=False)
single_image = images[0]  # Take the first image as an example

# Augment the single image
augmented_images, _ = augment([single_image], [0], num_aug=1)

# Extract DCT coefficients for the augmented image
Y_dct = jpeg_dct(augmented_images, 'Y')
U_dct = jpeg_dct(augmented_images, 'U')
V_dct = jpeg_dct(augmented_images, 'V')

# Load the pre-trained XGBoost model
xgboost_model = xgb.Booster()
xgboost_model.load_model('models/koniq/xgboost.json')

# Load the pre-trained feature extractors
Y_feature_extractor = HybridFeatures(channel='Y')
Y_feature_extractor.load('models/koniq')
U_feature_extractor = HybridFeatures(channel='U')
U_feature_extractor.load('models/koniq')
V_feature_extractor = HybridFeatures(channel='V')
V_feature_extractor.load('models/koniq')

# Extract features for the single image
y_features = Y_feature_extractor.transform(Y_dct)
u_features = U_feature_extractor.transform(U_dct)
v_features = V_feature_extractor.transform(V_dct)

# Concatenate the extracted features
features = np.concatenate([y_features, u_features, v_features], axis=-1)
num_features = features.shape[1]

# Define the input shape for the XGBoost model
xgboost_model_inputs = (1, num_features)

# Convert the XGBoost model to a PyTorch model
torch_xgboost = PyTorchConverter.from_xgboost(xgboost_model, inputs=xgboost_model_inputs)

# Save the converted PyTorch model
torch_xgboost.save('greenbiqa_pytorch_model.pth')