import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from LSH_random import *
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

image_folder = 'tests/101_ObjectCategories/butterfly'
feature_folder = 'tests/101_ObjectCategories/features'

def features_to_hash(feature_folder):
    feature_arrays = []
    for feature_file in os.listdir(feature_folder):
        feature_path = os.path.join(feature_folder, feature_file)
        feature = np.load(feature_path)
        feature_array = np.array(feature, dtype = np.float32)
        # feature_vectors = scaler.fit_transform(feature_array)
        print("feature_vectors:", feature_array.shape)

        feature_arrays.append(feature_array)

    features = lsh_random_projection(feature_arrays, feature_array.shape[1], 8)
  
# Só precisa chamar uma vez para extrair e salvar as features na pasta 'features'
# extract_and_save_features(image_folder, feature_folder)
features_to_hash(feature_folder)

# def features_to_hash(feature_folder):
#     feature_string = []
#     for feature_file in os.listdir(feature_folder)[:10]:
#         feature_path = os.path.join(feature_folder, feature_file)
#         feature = np.load(feature_path)
#         feature_array = np.array(feature, dtype = np.float32)
#         feature_array_as_string = list(map(str, feature_array.flatten())) 
#         feature_array_as_string = ' '.join(feature_array_as_string)
#         feature_string.append(feature_array_as_string)
#     features = lsh(feature_string, 8, 64)
#     print(features)
# # Só precisa chamar uma vez para extrair e salvar as features na pasta 'features'
# # extract_and_save_features(image_folder, feature_folder)
# features_to_hash(feature_folder)
