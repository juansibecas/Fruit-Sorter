# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 11:22:01 2020

@author: jpss8
"""

import skimage
from skimage import io
from skimage.color import rgb2gray
from skimage.exposure import histogram
from skimage.feature import greycomatrix
from skimage.feature import greycoprops
from skimage.filters.rank import otsu
from skimage.filters import threshold_otsu
from skimage.filters import threshold_multiotsu
from skimage.morphology import disk
from skimage.filters.rank import autolevel
from skimage.transform import resize
from skimage.filters import gaussian
from skimage import img_as_ubyte

import numpy as np

import math


class Image:
    
    def __init__(self, path, name):
        
        self.name = name
        
        self.height = 500
        self.width = 500
        
        self.original = io.imread(path)
        self.shape = self.original.shape
        
        if self.shape[0] > 1000 or self.shape[1] > 1000:   #resize images for faster analysis
            self.original = resize(self.original, (self.width, self.height))
            
        self.shape = self.original.shape  #image stylization
        self.gauss = gaussian(self.original)
        self.bilateral = skimage.restoration.denoise_bilateral(self.gauss, multichannel=True)
        self.grayscale = rgb2gray(self.gauss)
        self.gauss_2 = gaussian(self.grayscale)
        self.uint = img_as_ubyte(self.gauss_2)
        
        
        self.thresh = threshold_otsu(self.uint)    #otsu threshold
        self.binary = self.uint > self.thresh
        
        #self.multi_thresh = threshold_multiotsu(self.gauss_2, classes=2)  #multi threshold
        #self.binary = np.digitize(self.gauss_2, bins=self.multi_thresh)
        
        
        #self.local_thresh = skimage.filters.threshold_local(self.uint, 399) #adaptive
        #self.binary = self.uint > self.local_thresh
        
        
        self.mu = skimage.measure.moments_central(self.binary) #Hu moments
        self.nu = skimage.measure.moments_normalized(self.mu)
        self.hu = skimage.measure.moments_hu(self.nu)
        for index in range(len(self.hu)):
            self.hu[index] =  -1* math.copysign(1.0, self.hu[index]) * math.log10(abs(self.hu[index]))
            
        self.glcmatrix = greycomatrix(self.uint, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4]) 
        self.glcm_props = greycoprops(self.glcmatrix)                                   
        #GLCOM = gray level co ocurrence matrix
        #Texture properties Contrast, Correlation, Energy, Homogeneity
        
        
        #Perimeter and roundness
        self.perimeter_func = skimage.measure.perimeter(self.binary, 4)
        self.perimeter = 0
        self.area = 0
        self.calculate_perimeter_and_area()
        self.roundness = 2*math.sqrt(self.area * np.pi)/self.perimeter
        
        
        #RGB mean values
        self.histogram_1 = histogram(autolevel(self.original[:, :, 0], disk(5)))
        self.histogram_2 = histogram(autolevel(self.original[:, :, 1], disk(5)))
        self.histogram_3 = histogram(autolevel(self.original[:, :, 2], disk(5)))
        self.rgb_mean_1 = 0
        self.rgb_mean_2 = 0
        self.rgb_mean_3 = 0
        self.pixels_1 = 0
        self.pixels_2 = 0
        self.pixels_3 = 0
        for i in range(5,250):
            self.rgb_mean_1 += self.histogram_1[0][i]*i
            self.pixels_1 += self.histogram_1[0][i]

            self.rgb_mean_2 += self.histogram_2[0][i]*i
            self.pixels_2 += self.histogram_2[0][i]
 
            self.rgb_mean_3 += self.histogram_3[0][i]*i
            self.pixels_3 += self.histogram_3[0][i]
        
        self.rgb_mean_1 /= self.pixels_1
        self.rgb_mean_2 /= self.pixels_2
        self.rgb_mean_3 /= self.pixels_3

        #features vector creation. can choose which features to append  
        self.vector = []
        
        for i in range(2):
            self.vector.append(self.hu[i])
            
        #for i in range(4):
        #    self.vector.append(self.glcm_props[0][i])
        
        self.vector.append(self.roundness)
        
        #self.vector.append(self.perimeter_func)
        
        self.vector.append(self.rgb_mean_1)
        self.vector.append(self.rgb_mean_2)
        self.vector.append(self.rgb_mean_3)
        
        
    def calculate_perimeter_and_area(self):
        for i in range(len(self.binary)):
            for j in range(len(self.binary[0])):
                if self.binary[i][j]:
                    if False in self.check_surroundings(self.binary, i, j):
                        self.perimeter +=1
                else:
                    self.area +=1
        
                            
        
    def check_surroundings(self, array, i, j):
        surroundings = []
        for m in range(-1, 2):
            for n in range(-1, 2):
                if i+m == j+n: continue
                try:    
                    surroundings.append(array[i+m][j+n])
                except:
                    continue
        return surroundings
