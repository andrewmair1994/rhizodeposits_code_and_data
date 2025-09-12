# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 09:21:30 2025

@author: andre
"""

# This is a FEniCS script that works out the rooting depth of each root 
# system.

###############################################################################
# Importing necessary libaries
###############################################################################

import numpy as np

###########################################################################
# Importing the crootbox root data of all systems into one array
###########################################################################

# Importing each root system.
segments = np.loadtxt('data/trigo6daysformatted.txt')
segments = np.append(segments, np.loadtxt('data/trigo15daysformatted.txt'), axis = 0)
segments = np.append(segments, np.loadtxt('data/trigo30daysformatted.txt'), axis = 0)

###########################################################################
# Defining the soil domain that contains all root systems
# Meshing the domain and then saving it out
###########################################################################

# Finding the lowest x_1 value of a segment's end in the architecture.
minx10 = np.amin(segments[:, 0]) 
minx14 = np.amin(segments[:, 4])
minx1 = np.min(np.array([minx10, minx14]))

# Finding the highest x_1 value of a segments's end in the architecture.
maxx10 = np.amax(segments[:, 0])
maxx14 = np.amax(segments[:, 4])
maxx1 = np.amax(np.array([maxx10, maxx14]))

# Finding the lowest x_2 value of a segment's end in the architecture.
minx21 = np.amin(segments[:, 1]) 
minx25 = np.amin(segments[:, 5])
minx2 = np.amin(np.array([minx21, minx25]))

# Finding the highest x_2 value of a segments's end in the architecture.
maxx21 = np.amax(segments[:, 1])
maxx25 = np.amax(segments[:, 5])
maxx2 = np.amax(np.array([maxx21, maxx25]))

# Finding the lowest x_3 value of a segment's end in the architecture.
minx32 = np.amin(segments[:, 2]) 
minx36 = np.amin(segments[:, 6])
minx3 = np.amin(np.array([minx32, minx36]))

# Finding the highest x_3 value of a segments's end in the architecture.
maxx32 = np.amax(segments[:, 2])
maxx36 = np.amax(segments[:, 6])
maxx3 = np.amax(np.array([maxx32, maxx36]))

# print("x_1 minimum =", minx1)
# print("x_1 maximum =", maxx1)
# print("x_2 minimum =", minx2)
# print("x_2 maximum =", maxx2)

# Saving original x_3 limits.
minx3_old = minx3
maxx3_old = maxx3

# If the highest x_3 value is above zero then the whole architecture is
# shifted down by this value.
if maxx3 > 0:
    segments[:, 2] -= maxx3
    segments[:, 6] -= maxx3
    
# Finding the new lowest x_3 value of a segment's end in the architecture.
minx32 = np.amin(segments[:, 2]) 
minx36 = np.amin(segments[:, 6])
minx3 = np.amin(np.array([minx32, minx36]))
print('Domain depth', minx3) 

###############################################################################
# Shifting the segement values in each root system separately
###############################################################################

segments_6day = np.loadtxt('data/trigo6daysformatted.txt')
segments_15day = np.loadtxt('data/trigo15daysformatted.txt')
segments_30day = np.loadtxt('data/trigo30daysformatted.txt')

# Shifting
segments_6day[:, 2] -= maxx3
segments_6day[:, 6] -= maxx3
segments_15day[:, 2] -= maxx3
segments_15day[:, 6] -= maxx3
segments_30day[:, 2] -= maxx3
segments_30day[:, 6] -= maxx3

###############################################################################
# Finding the lowest segment in each root system separately
###############################################################################

# 6day old root system
minx32_6day = np.amin(segments_6day[:, 2]) 
minx36_6day = np.amin(segments_6day[:, 6])
minx3_6day = np.amin(np.array([minx32_6day, minx36_6day]))
print('Rooting depth 6 day old system', minx3_6day)
# np.savetxt('data/trigo6days_rooting_depth.txt', np.array([minx3_6day])) 

# 15day old root system
minx32_15day = np.amin(segments_15day[:, 2]) 
minx36_15day = np.amin(segments_15day[:, 6])
minx3_15day = np.amin(np.array([minx32_15day, minx36_15day])) 
print('Rooting depth 15 day old system', minx3_15day)
# np.savetxt('data/trigo15days_rooting_depth.txt', np.array([minx3_15day])) 

# 30day old root system
minx32_30day = np.amin(segments_30day[:, 2]) 
minx36_30day = np.amin(segments_30day[:, 6])
minx3_30day = np.amin(np.array([minx32_30day, minx36_30day]))
print('Rooting depth 30 day old system', minx3_30day)
# np.savetxt('data/trigo30days_rooting_depth.txt', np.array([minx3_30day]))  