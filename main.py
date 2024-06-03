import os
import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
import matplotlib.pyplot as plt

#usando imagens do dataset 101_ObjectCategories, butterfly, https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp/view?usp=drive_link

image_folder = '101_ObjectCategories/butterfly'
feature_folder = 'features'

# Usando modelo pré treinado CNN (VGG16)
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)  # usando layer 'block5_conv3' 

# Colocando no modelo para processar as imagens no padrao do VGG16
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Extraindo e salvando as features
def extract_and_save_features(image_folder, feature_folder):
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder)

    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        img_array = preprocess_image(img_path)
        features = model.predict(img_array)  
        feature_path = os.path.join(feature_folder, os.path.splitext(img_name)[0] + '.npy')
        np.save(feature_path, features) 

def get_images_similar_to(image_path, feature_folder):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # features = model.predict(img_array)
    feature_path = os.path.join(feature_folder, image_path.split('/')[-1].split('.')[0] + '.npy')
    feature = np.load(feature_path)

    feature_distances = []

    for feature_file_compare in os.listdir(feature_folder):
        feature_path_compare = os.path.join(feature_folder, feature_file_compare)
        feature_compare = np.load(feature_path_compare)
        # Linalg = distancia euclidiana
        distances = np.linalg.norm(feature_compare - feature)
        print("distances:", distances)
        feature_distances.append(distances)

    print("feature_distances:", feature_distances)
    print("len(feature_distances):", len(feature_distances))
    # para organizar entre as 5 imagens mais similares, ou seja, com menor distância (argsort da os indices)
    similar_indices = np.argsort(feature_distances)[0:5]
    print("similar_indices:", similar_indices)

    image_dir = os.path.dirname(image_path)
    print(image_dir)
    count = 0
    plt.subplot(1, 1, 1)
    img = image.load_img(image_path, target_size=(224, 224))
    plt.imshow(img)
    plt.axis('off')

    for i in similar_indices:
        plt.subplot(5, 5, count + 1)
        count += 1
        img = image.load_img(os.path.join(image_dir, os.listdir(image_dir)[i]), target_size=(224, 224))
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(f'Images similar to {os.path.basename(image_path)}', fontsize=20)
    plt.show()

# Só precisa chamar uma vez para extrair e salvar as features na pasta 'features'
# extract_and_save_features(image_folder, feature_folder)

get_images_similar_to('101_ObjectCategories/butterfly/image_0001.jpg', feature_folder)