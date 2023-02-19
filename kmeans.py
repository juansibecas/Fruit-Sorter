# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:47:47 2020

@author: jpss8
"""

import math
import random
from statistics import mode
from copy import copy


class Kmeans:
    """
    Kmeans initializer with predict method.
    """
    def __init__(self, n_clusters, data):
        self.data = data
        self.n_clusters = n_clusters
        self.data_len = len(data)
        self.vector_len = len(data[0]['vector'])

        self.clusters = []
        
        # initialize random clusters seed
        for cluster in range(self.n_clusters):
            self.clusters.append({'cluster seed':[], 'cluster points list':[], 'old cluster centroid':[], 'new cluster centroid':[], 'name':0})
            for j in range(self.vector_len):
                self.clusters[cluster]['cluster seed'].append(random.uniform(0, 1))  # all data already normalized, all values will stay between 0 and 1
                self.clusters[cluster]['old cluster centroid'].append(self.clusters[cluster]['cluster seed'][j])
        
        max_it = 100
        self.cluster_delta = 20
        tol = 0.01
        temp = 0
        temp_2 = 0
        it = 0
        
        distance_list = []
        name_list = []
        
        while self.cluster_delta > tol and it < max_it:  # iterates to find optimum clusters' centroid positions
            
            for data_index in range(self.data_len):   # calculates distance to each cluster for each point. assigns each point to its closest cluster
                for cluster_index in range(self.n_clusters):
                    for vector_index in range(self.vector_len):
                        temp += pow(self.data[data_index]['vector'][vector_index] - self.clusters[cluster_index]['old cluster centroid'][vector_index], 2)
                    distance_list.append([cluster_index, copy(temp)])
                    temp = 0
                distance_list.sort(reverse=False, key=sort_table_by_distance)
                cluster_number = copy(distance_list[0][0])   # lower distance cluster's index
                self.clusters[cluster_number]['cluster points list'].append({'name':self.data[data_index]['name'], 'vector':self.data[data_index]['vector']})
                distance_list.clear()
            
            for cluster_index in range(self.n_clusters):  # calculates new centroids (mean of each dimension for all points the cluster was assigned)
                for vector_index in range(self.vector_len):
                    points_len = len(self.clusters[cluster_index]['cluster points list'])
                    for point_index in range(points_len): 
                        temp += self.clusters[cluster_index]['cluster points list'][point_index]['vector'][vector_index]
                    try:
                        temp /= points_len
                    except:  # many times a cluster isnt assigned any datapoint
                        temp = 0
                    self.clusters[cluster_index]['new cluster centroid'].append(copy(temp))
                    temp = 0

            for cluster_index in range(self.n_clusters):  # calculate cluster_delta (distance between new and old centroids)
                for vector_index in range(self.vector_len):
                    temp += pow(self.clusters[cluster_index]['new cluster centroid'][vector_index]-self.clusters[cluster_index]['old cluster centroid'][vector_index], 2)
                temp_2 += copy(math.sqrt(temp))
                temp = 0
            self.cluster_delta = copy(temp_2)
            temp_2 = 0

            for cluster_index in range(self.n_clusters):  # assign name to each cluster (mode of all the datapoints names it contains)
                for point_index in range(len(self.clusters[cluster_index]['cluster points list'])):
                    name_list.append(self.clusters[cluster_index]['cluster points list'][point_index]['name'])
                try:
                    self.clusters[cluster_index]['name'] = mode(copy(name_list))
                except:
                    self.clusters[cluster_index]['name'] = 'None'
                name_list.clear()

            for cluster_index in range(self.n_clusters):  # update old centroid and clear new centroid
                self.clusters[cluster_index]['old cluster centroid'] = copy(self.clusters[cluster_index]['new cluster centroid'])
                self.clusters[cluster_index]['new cluster centroid'].clear()
                
            it += 1

    def predict(self, query):
        """
        Returns a category given a query (vector of image properties)
        """
        distance = 0    
        distance_list = []
        for cluster_index in range(self.n_clusters):  # distance to each cluster for query point. returns the query's closest cluster's name
            for vector_index in range(self.vector_len):
                centroid_value = self.clusters[cluster_index]['old cluster centroid'][vector_index]
                query_value = query[vector_index]
                distance += math.pow(centroid_value - query_value, 2)
            distance_list.append( [self.clusters[cluster_index]['name'], math.sqrt(copy(distance)) ] )
            distance = 0
        distance_list.sort(reverse=False, key=sort_table_by_distance) 
        name = distance_list[0][0]
        return name
    
    def print_cluster_delta(self):
        print(self.cluster_delta)


def sort_table_by_distance(name_and_distance_list):
    return name_and_distance_list[1]
