# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 17:47:44 2020

@author: jpss8
"""

import math
from statistics import mode      


def knn(data, query, k):
    """
    Returns the mode of the k nearest neighbours to the query in the vector(1xN) space(NxN)
    """
    data_len = len(data)
    distance_list = []
    name_list = []

    for data_index in range(data_len):  # appends [name, distance] to a list for each data point
        distance_list.append(calculate_euclidean_distance_and_assign_name(data, query, data_index))  
        
    distance_list.sort(reverse=False, key=sort_table_by_distance)    
    for index in range(k):  # collects the first k names
        name_list.append(distance_list[index][0])
        
    return mode(name_list)


def calculate_euclidean_distance_and_assign_name(data, query, point_index):
    vector_len = len(data[0]['vector'])
    distance = 0
       
    for vector_index in range(vector_len):
        distance += math.pow(data[point_index]['vector'][vector_index] - query[vector_index], 2)
    distance = math.sqrt(distance)
    name = data[point_index]['name']

    return [name, distance]


def sort_table_by_distance(name_and_distance_list):
    return name_and_distance_list[1]