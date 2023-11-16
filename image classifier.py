# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 10:56:02 2020

@author: jpss8
"""

import matplotlib.pyplot as plt
from image import Image
from properties import Property
from knn import knn
from kmeans import Kmeans
from copy import copy
from warnings import filterwarnings


def run():
    filterwarnings('ignore')
    banana_images = []
    tomato_images = []
    orange_images = []
    lemon_images = []

    banana_properties = {'txt props': [], 'hu moments': [], 'perimeter': 0}
    tomato_properties = {'txt props': [], 'hu moments': [], 'perimeter': 0}
    orange_properties = {'txt props': [], 'hu moments': [], 'perimeter': 0}
    lemon_properties = {'txt props': [], 'hu moments': [], 'perimeter': 0}

    banana = {'images': banana_images, 'properties': banana_properties, 'color': 'green'}
    tomato = {'images': tomato_images, 'properties': tomato_properties, 'color': 'red'}
    orange = {'images': orange_images, 'properties': orange_properties, 'color': 'orange'}
    lemon = {'images': lemon_images, 'properties': lemon_properties, 'color': 'yellow'}

    fruits = {'banana': banana, 'tomato': tomato, 'orange': orange, 'lemon': lemon}

    testing_images = []

    # path to the image folder
    path = '/frutas/'

    training_images = 6
    # testing = total - training
    total_images = 16
    names = ['banana', 'tomato', 'orange', 'lemon']
    
    offset = 0  # used for algorithm validation
    for index in range(training_images):
        
        name_banana = 'banana' + str(index) + '.jpg'
        
        banana_images.append(Image(path + name_banana, names[0]))
        
        name_tomato = 'tomate' + str(index) + '.jpg'
        tomato_images.append(Image(path + name_tomato, names[1]))
        
        name_orange = 'naranja' + str(index) + '.jpg'
        orange_images.append(Image(path + name_orange, names[2]))
        
        name_lemon = 'limon' + str(index) + '.jpg'
        lemon_images.append(Image(path + name_lemon, names[3]))
        print("Processed ", (index + 1)*4, "/", training_images*4, " training images")
        
    for index in range(training_images, total_images):
        
        name_banana = 'banana' + str(index-offset) + '.jpg'
        testing_images.append(Image(path + name_banana, names[0]))
 
        name_tomato = 'tomate' + str(index-offset) + '.jpg'
        testing_images.append(Image(path + name_tomato, names[1]))

        name_orange = 'naranja' + str(index-offset) + '.jpg'
        testing_images.append(Image(path + name_orange, names[2]))
  
        name_lemon = 'limon' + str(index-offset) + '.jpg'
        testing_images.append(Image(path + name_lemon, names[3]))
        
        print("Processed ", (index - offset - training_images + 1)*4, "/", (total_images - offset - training_images)*4, " testing images")
        
    print("All images processed")
    
    hu_length = 7
    hu = []
    glcm_props_length = 4
    glcm_props = []
    perimeter_length = 1
    perimeter = []
    roundness_length = 1
    roundness = []
    rgb_length = 1
    rgb_mean_1 = []
    rgb_mean_2 = []
    rgb_mean_3 = []
      
    for fruit_index, fruit_name in enumerate(fruits):  # data collection for stats
        
        for prop_index in range(hu_length):  # Hu Moments
            for image_index in range(training_images):   
                hu.append(fruits[fruit_name]['images'][image_index].hu[prop_index])
            fruits[fruit_name]['properties']['hu moments'].append(Property(copy(hu)))
            hu.clear()
            
        for prop_index in range(glcm_props_length):  # Gray Level Co Ocurrence Matrix Properties
            for image_index in range(training_images):   
                glcm_props.append(fruits[fruit_name]['images'][image_index].glcm_props[0][prop_index])
            fruits[fruit_name]['properties']['txt props'].append(Property(copy(glcm_props)))
            glcm_props.clear()
            
        for prop_index in range(perimeter_length):  # Perimeter
            for image_index in range(training_images):   
                perimeter.append(fruits[fruit_name]['images'][image_index].perimeter_func)
            fruits[fruit_name]['properties']['perimeter']=Property(copy(perimeter))
            perimeter.clear()
            
        for prop_index in range(roundness_length):  # Roundness
            for image_index in range(training_images):   
                roundness.append(fruits[fruit_name]['images'][image_index].roundness)
            fruits[fruit_name]['properties']['roundness']=Property(copy(roundness))
            roundness.clear()
            
        for prop_index in range(rgb_length):  # rgb mean by channel
            for image_index in range(training_images):   
                rgb_mean_1.append(fruits[fruit_name]['images'][image_index].rgb_mean_1)
                rgb_mean_2.append(fruits[fruit_name]['images'][image_index].rgb_mean_2)
                rgb_mean_3.append(fruits[fruit_name]['images'][image_index].rgb_mean_3)
            fruits[fruit_name]['properties']['rgb mean 1']=Property(copy(rgb_mean_1))
            fruits[fruit_name]['properties']['rgb mean 2']=Property(copy(rgb_mean_2))
            fruits[fruit_name]['properties']['rgb mean 3']=Property(copy(rgb_mean_3))
            rgb_mean_1.clear()
            rgb_mean_2.clear()
            rgb_mean_3.clear()

    # filters plot
    for fruit_name in fruits:
        for index in range(training_images):
            fig, axes = plt.subplots(3, 2, figsize=(8, 8))
            ax = axes.ravel()
            
            ax[0].imshow(fruits[fruit_name]['images'][index].original)
            ax[0].set_title("Original")
            
            ax[1].imshow(fruits[fruit_name]['images'][index].gauss)
            ax[1].set_title("Bilateral")
            
            ax[2].imshow(fruits[fruit_name]['images'][index].bilateral)
            ax[2].set_title("Gauss")

            ax[3].imshow(fruits[fruit_name]['images'][index].grayscale, cmap=plt.cm.gray)
            ax[3].set_title("Grayscale")
            
            ax[4].imshow(fruits[fruit_name]['images'][index].gauss_2, cmap=plt.cm.gray)
            ax[4].set_title("Gauss")

            ax[5].imshow(fruits[fruit_name]['images'][index].binary, cmap=plt.cm.binary)
            ax[5].set_title("Binary Otsu")

            fig.tight_layout()
            plt.show()
    
    """
    for fruit_name in fruits: #stats print
        print(fruit_name)
        
        print("Hu Moments")
        for prop_index in range(hu_length):
            print(fruits[fruit_name]['properties']['hu moments'][prop_index].mean, " ", 100*fruits[fruit_name]['properties']['hu moments'][prop_index].rstd)
        
        print("Texture Properties")
        for prop_index in range(glcm_props_length):
            print(fruits[fruit_name]['properties']['txt props'][prop_index].mean, " ", 100*fruits[fruit_name]['properties']['txt props'][prop_index].rstd)
        
        print("Roundness")
        print(fruits[fruit_name]['properties']['roundness'].mean, " ", 100*fruits[fruit_name]['properties']['roundness'].rstd)
        
        print("Perimeter")
        print(fruits[fruit_name]['properties']['perimeter'].mean, " ", 100*fruits[fruit_name]['properties']['perimeter'].rstd)
        
        print("RGB Means")
        print(fruits[fruit_name]['properties']['rgb_mean_1'].mean, " ", 100*fruits[fruit_name]['properties']['rgb_mean_1'].rstd)
        print(fruits[fruit_name]['properties']['rgb_mean_2'].mean, " ", 100*fruits[fruit_name]['properties']['rgb_mean_2'].rstd)
        print(fruits[fruit_name]['properties']['rgb_mean_3'].mean, " ", 100*fruits[fruit_name]['properties']['rgb_mean_3'].rstd)
    """
             
    data = []
    
    for fruit_name in fruits:  # data collection and normalization for knn and kmeans
        for image_index in range(training_images):
            vector = fruits[fruit_name]['images'][image_index].vector
            name = fruit_name
            data.append({'name':copy(name), 'vector':copy(vector)})

    normalized_data, max_and_min_list = normalize_database(data)

    # kmeans classes initialization
    kmeans_list = []
    kmeans_amount = 5
    for clusters in range(kmeans_amount):
        kmeans_list.append(Kmeans((clusters+1)*4, normalized_data))
        print("Kmeans", (clusters+1)*4, " initialized")
    
    # knn and kmeans execution
    knn_amount = 5
    knn_hits = [0] * knn_amount
    kmeans_hits = [0] * kmeans_amount
    tries = 0
    for image in testing_images:
        query = image.vector
        normalized_query = normalize_vector(query, max_and_min_list)
        print("Image is: ", image.name)
        # knn predictions
        for knn_index in range(knn_amount):
            knn_prediction = knn(normalized_data, normalized_query, (knn_index*2)+3)
            print("Knn", (knn_index*2)+3 ," says: ", knn_prediction)
            if image.name == knn_prediction:
                knn_hits[knn_index] += 1
        # kmeans predictions
        for kmeans_index in range (kmeans_amount):
            kmeans_prediction = kmeans_list[kmeans_index].predict(normalized_query)
            print("Kmeans", (kmeans_index+1)*4, " says: ", kmeans_prediction)
            if image.name == kmeans_prediction:
                kmeans_hits[kmeans_index] += 1
        print("\n")
        tries += 1
        
    # prints results
    for index, hits in enumerate(knn_hits):
        print("KNN", 2*index+3, " Accuracy: ", 100*hits/tries)
    for index, hits in enumerate(kmeans_hits):
        print("Kmeans", 4*(index+1), " Accuracy: ", 100*hits/tries)

    return fruits


def normalize(value, max_value, min_value):
    return (value - min_value)/(max_value - min_value)


def normalize_vector(vector, max_and_min_list):
    vector_len = len(vector)
    for vector_index in range(vector_len):
        vector[vector_index] = normalize(vector[vector_index], max(max_and_min_list[vector_index]), min(max_and_min_list[vector_index]))
    
    return vector


def normalize_database(data):
    """
    Normalizes the full database and returns a max and min
    list for each component for further vector normalizations.
    """
    vector_len = len(data[0]['vector'])
    data_len = len(data)
    max_and_min_list = []
    for vector_index in range(vector_len):
        max_and_min_list.append([])
        for data_index in range(data_len):
            max_and_min_list[vector_index].append(copy(data[data_index]['vector'][vector_index]))
            
    for data_index in range(data_len):
        for vector_index in range(vector_len):
            data[data_index]['vector'][vector_index] = normalize(data[data_index]['vector'][vector_index], max(max_and_min_list[vector_index]), min(max_and_min_list[vector_index]))
    return data, max_and_min_list


if __name__ == '__main__':
    fruits = run()