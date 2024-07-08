import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

import imagehash

"""
Implementação baseada no artigo:
- Perceptual Hashing using Convolutional Neural
Networks for Large Scale Reverse Image Search
Mathieu Gaillard

- https://www.phash.org/docs/pubs/thesis_zauner.pdf

- https://www.geeksforgeeks.org/discrete-cosine-transform-algorithm-program/
"""

image_folder = 'tests/101_ObjectCategories/butterfly'
image_folder_perceptual = 'perceptualHashingTests/images'

def DCT_image(img_array):
    dct_values = [[0 for i in range(img_array.shape[1])] for j in range(img_array.shape[0])]
    
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            ci = (2 ** (1/2)) / (img_array.shape[0] ** (1/2))
            cj = (2 ** (1/2)) / (img_array.shape[1] ** (1/2))
            if i == 0:
                ci = 1 / (img_array.shape[0] ** (1/2))
            if j == 0:
                cj = 1 / (img_array.shape[1] ** (1/2))

            somatorio = 0
            for k in range(img_array.shape[0]):
                for l in range(img_array.shape[1]):
                    somakl = img_array[k][l] * np.cos((2 * k + 1) * i * np.pi / (2 * img_array.shape[0])) * np.cos((2 * l + 1) * j * np.pi / (2 * img_array.shape[1]))
                    somatorio += somakl

            dct_values[i][j] = ci * cj * somatorio

    return np.array(dct_values)

def DCT_Perceptual_Hashing_Implementado(img_path, img_name, resize_size=32, dct_block_size=(8,8)):
        image = Image.open(img_path)

        # o resultado é convertido para preto e branco
        img_preto_branco = image.convert('L').resize((resize_size, resize_size), Image.Resampling.LANCZOS)
        # depois é borrado com boxblur para eliminar ruido
        img_com_blur = img_preto_branco.filter(ImageFilter.BoxBlur(radius= 7))
        # transformando para array
        img_array = np.asarray(img_com_blur, dtype = np.float32)
        print(img_array)
        # utilizando a DCT do OpenCV
        img_dct = cv.dct(img_array)

        # * Implementado manualmente dando resultados identicos a biblioteca
        #img_dct = DCT_image(img_array)
        # plt.imshow(img_dct)
        # plt.show()
        print(img_dct)

        # cortando o bloco de 8x8
        img_dct = img_dct[0:dct_block_size[0], 0:dct_block_size[0]]
        
        # calculando a mediana
        mediana = np.median(img_dct)

        hash_binario = 0

        img_dct = img_dct.flatten()
        for i in range(img_dct.shape[0]):
            hash_binario <<= 1
            if img_dct[i].flatten()[0] > mediana:
                hash_binario |= 0x1
        hash_binario &= 0xFFFFFFFFFFFFFFFF
   
        # retornando o hash
        return hash_binario

def DCT_Perceptual_Hashing_LibImageHash(img_path, img_name, resize_size=32, dct_block_size=(8,8)):
    img = Image.open(img_path)
    hash = imagehash.phash(img)
    # hex to decimal
    hex_hash = str(hash)
    hash_valores = int(hex_hash, 16)   
    return hash_valores

def DCT_Perceptual_Hashing(image_folder, resize_size=32, dct_block_size=(8,8)):
    hash_list = []
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        hash_binario = DCT_Perceptual_Hashing_Implementado(img_path, img_name, resize_size, dct_block_size)
        print(hash_binario)
        hash_list.append(hash_binario)

    # for i in range(len(hash_list)):
    #     print(f'{hash_list[i]:064b}')
    #     # to hex
    #     print(f'{hash_list[i]:016x}')

    return hash_list

"""
Hamming Distance

Valores entre 0 e 1, mais proximo de 0 significa que as imagens sao mais similares
"""
def hamming_distance(hash_referencia, hash_list):
    hamming_list = []
    index_img = 0
    for hash in hash_list:
        # print(hash)
        # print(hash_referencia)
        # hamming = sum(c1 != c2 for c1, c2 in zip(hash_referencia, hash))
        hamming = bin(hash ^ hash_referencia).count('1')
        print(hamming)
        print(len(bin(hash)) - 2)
        hamming_list.append([hamming/(len(bin(hash)) - 2), index_img])
        index_img += 1
    return hamming_list


def gera_imagens_similares_experimento():
    # imagens para testar perceptual hashing de imagens com diferentes operações
    imagem_selecionada = os.listdir(image_folder_perceptual)[0]

    imagem_teste = Image.open(os.path.join(image_folder_perceptual, imagem_selecionada))

    # operacao blur

    imagem_blur = imagem_teste.filter(ImageFilter.BLUR)

    imagem_blur.save(os.path.join(image_folder_perceptual, "image_test_blur.jpg"))

    # operacao preto_branco

    imagem_preto_branco = imagem_teste.convert('L')

    imagem_preto_branco.save(os.path.join(image_folder_perceptual, "image_test_preto_branco.jpg"))

    # operacao resize

    imagem_resize = imagem_teste.resize((128, 128))

    imagem_resize.save(os.path.join(image_folder_perceptual, "image_test_resize.jpg"))

    # operacao compressao

    imagem_compressao = imagem_teste.filter(ImageFilter.SHARPEN)

    imagem_compressao.save(os.path.join(image_folder_perceptual, "image_test_compressao.jpg"))

    # operacao rotacao

    imagem_rotacao = imagem_teste.rotate(90)

    imagem_rotacao.save(os.path.join(image_folder_perceptual, "image_test_rotacao.jpg"))

    # operacao flip

    imagem_flip = imagem_teste.transpose(Image.FLIP_LEFT_RIGHT)

    imagem_flip.save(os.path.join(image_folder_perceptual, "image_test_flip.jpg"))

    # operacao crop

    imagem_crop = imagem_teste.crop((0, 0, imagem_teste.size[0] // 2, imagem_teste.size[0] // 2))

    imagem_crop.save(os.path.join(image_folder_perceptual, "image_test_crop.jpg"))

def hash_int_to_hex(hash_int):
    return hex(hash_int)


def experimento_efetividade_operacoes_imagens():
    gera_imagens_similares_experimento()

    lista_imagens = os.listdir(image_folder_perceptual)
    hash_list = []
    for img_name in lista_imagens:
        img_path = os.path.join(image_folder_perceptual, img_name)
        hash_binario = DCT_Perceptual_Hashing_Implementado(img_path, img_name, resize_size=32, dct_block_size=(8,8))
        hash_list.append(hash_binario)
    
    hamming_dist_list = hamming_distance(hash_list[0], hash_list[0:])
    # print(f'Considerando hash com {len(bin(hash_list[0])) - 2} bits, as imagens com as mesmas operações tem distancias de hamming de: {hamming_dist_list}')
    # for i in range(len(hamming_dist_list)):
    #     print(f' Imagem: {lista_imagens[i]} : Distancia de Hamming: {hamming_dist_list[i][0]} | hash: {hash_int_to_hex(hash_list[i])}')
    # print(hamming_dist_list)

    hamming_dist_list.sort()
    print(hamming_dist_list)
    imagem_selecionada = os.listdir(image_folder_perceptual)[0]
    img = Image.open(os.path.join(image_folder_perceptual, imagem_selecionada))

    plt.subplot(1, 1, 1)
    plt.imshow(img)
    plt.axis('off')
    count = 0
    print(os.listdir(image_folder_perceptual))
    fixed_radius_nearest_neighbors(hash_list[2], hash_list, 1)
    for i in range(len(hamming_dist_list)):
        plt.subplot(9, 9, count + 1)
        count += 1
        img = Image.open(os.path.join(image_folder_perceptual, lista_imagens[hamming_dist_list[i][1]]))
        print(f' Imagem: {lista_imagens[hamming_dist_list[i][1]]} : Distancia de Hamming: {hamming_dist_list[i][0]} | hash: {hash_int_to_hex(hash_list[hamming_dist_list[i][1]])}')
        plt.imshow(img, )
        plt.axis('off')

    plt.show()

def fixed_radius_nearest_neighbors(y, z, radius):
    print(y)
    print(z)
    # for i in range(len(y)):
        
    return True

experimento_efetividade_operacoes_imagens()

def experimento_busca_imagem_similar():
    hash_list = DCT_Perceptual_Hashing(image_folder, resize_size=32, dct_block_size=(8,8))
    hamming_dist_list = hamming_distance(hash_list[14], hash_list)
    hamming_dist_list.sort()

    imagem_selecionada = os.listdir(image_folder)[14]
    img = Image.open(os.path.join(image_folder, imagem_selecionada))

    plt.subplot(1, 1, 1)
    plt.imshow(img)
    plt.axis('off')
    count = 0
    print(os.listdir(image_folder))
    for i in hamming_dist_list[:5]:
        plt.subplot(5, 5, count + 1)
        count += 1
        print(i[1])
        img = Image.open(os.path.join(image_folder, os.listdir(image_folder)[i[1]]))
        plt.imshow(img, )
        plt.axis('off')


    plt.show()

