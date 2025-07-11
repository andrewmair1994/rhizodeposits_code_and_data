# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:37:09 2024

@author: andre
"""

# This is a FEniCS script that converts CRootBox root system data into a form
# that can be used to generate root length and surface area density functions.
# The code also creates a mesh in FEniCS for the smallest soil domain in which 
# the 6, 15 and 30 day old root systems can all fit.


###############################################################################
# Importing necessary libaries
###############################################################################

from fenics import *
from dolfin import *
import numpy as np

###############################################################################
# Defining a function that formats raw CrootBox data.
###############################################################################

def data_formatter(name):
    """
    

    Parameters
    ----------
    name : String
        Root system data to be formatted.

    Returns
    -------
    segments : np.array(,)
        Positional and dimensional information on segments of root system
        that is compatible with density construction method.
    biomass : Float
        Volume occupied by root system

    """
    
    # Importing segment array and manipulating into required format.
    data = np.loadtxt(fname = f'data/{name}.txt', skiprows = 1)
    
    # Removing any segments with zero length.
    if np.min(data[:, 10]) == 0:
        print("Error: some segments have zero length by crootbox measurement")
        
        rows_with_zero_lengths = np.argwhere(data[:,10] == 0)
        print("rows with zero length segments =", rows_with_zero_lengths)
        
        print("Shape of data prior to removal =", np.shape(data))

        print("Removing rows with zero lengths")
        data = data[data[:, 10] > 0]
        print("New minimum segment length =", np.min(data[:, 10]))
        
        print("Shape of data after removing segments of zero length =", np.shape(data))
        
    segments = np.zeros([len(data), 8])
    segments[:, 0:3] = data[:, 3:6] 
    segments[:, 3] = data[:, 9]
    segments[:, 7] = data[:, 9]  
    segments[:, 4:7] = data[:, 6:9]

    # Creating vector with length of each segment computed as the 
    # euclidean distance between the 2 end points.
    segments_lengths = np.sqrt((segments[:, 0] - segments[:, 4])**2 
                                + (segments[:, 1] - segments[:, 5])**2  
                                + (segments[:, 2] - segments[:, 6])**2)
        
    # Removing any segments where the euclidean distance between points a and b
    # is 0.
    if np.min(segments_lengths) == 0:
        print("Error: some segments have zero length by euclidean distance")
        print("Size before row removal =", np.shape(segments))
        
        zero_lengths = np.argwhere(segments_lengths == 0)
        print("rows with zero length segments =", zero_lengths)
        
        # Removing segments with zero Euclidean distance from the array
        segments = np.delete(segments, zero_lengths, axis = 0)
        print("Size after row removal =", np.shape(segments))    
        
    # Saving out segment array for this individual root system
    np.savetxt(f'data/{name}formatted.txt', segments)
    
    # Creating vector with length of each segment computed as the 
    # euclidean distance between the 2 end points. Again but with segments of
    # zero length removed.
    segments_lengths = np.sqrt((segments[:, 0] - segments[:, 4])**2 
                                + (segments[:, 1] - segments[:, 5])**2  
                                + (segments[:, 2] - segments[:, 6])**2)
    
    # Vector containing the radius of end one of each segment.
    proximal_radii = segments[:, 3]
        
    # Vector containing the radius of end two of each segment. 
    distal_radii = segments[:, 7]
    
    # Defining pi.
    pi = np.pi
    
    # Vector containing the volume of each segment.
    segments_volumes = (segments_lengths*pi/3)*((proximal_radii**2 + proximal_radii*distal_radii + distal_radii**2))
    
    # Vector containing the lateral surface area of each segment.
    segments_lateral_areas = pi*(proximal_radii + distal_radii)*np.sqrt(segments_lengths**2 + (proximal_radii - distal_radii)**2)
    
    # surface ares of the root system within the soil column.
    architecture_surface_area = np.sum(segments_lateral_areas)
    print(f"Total architecture surface area of {name}=", architecture_surface_area)
    
    # Total root length.
    total_root_length = np.sum(segments_lengths)
    print(f"Total root length of {name}=", total_root_length)
    
    # Computing root biomass.
    biomass = np.sum(segments_volumes)
    print(f"Total root biomass of {name}=", biomass)
    
    

    # Total number of segments in individual system.
    N_segments = len(segments[:, 0])
    np.savetxt(f'data/{name}_number.txt', np.array([N_segments]))
    
    # Saving out root architecture length
    np.savetxt(f'data/{name}_total_length.txt', np.array([total_root_length]))
    
    # Saving out root architecture surface area
    np.savetxt(f'data/{name}_architecture_surface_area.txt', np.array([architecture_surface_area]))
    
    # Saving out total root biomass.
    np.savetxt(f'data/{name}_biomass.txt', np.array([biomass]))
    
    return segments

###############################################################################
# Applying data formatter function
###############################################################################

trigo6days = data_formatter('trigo6days')
trigo15days = data_formatter('trigo15days')
trigo30days = data_formatter('trigo30days')

###############################################################################
# Defining a function which generates a global mesh defined by the extremal
# points of the 3 root systems.
###############################################################################

def mesher_global(refinement):
    
    '''
    

    Parameters
    ----------
    refinement : Integer
        Level of mesh refinement. The default is 32.

    Returns
    -------
    mesh : dolfin mesh
        The mesh that has been created.

    '''
    
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
    
    print("x_1 minimum =", minx1)
    print("x_1 maximum =", maxx1)
    print("x_2 minimum =", minx2)
    print("x_2 maximum =", maxx2)
    
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
    
    # Finding the new highest x_3 value of a segments's end in the architecture.
    maxx32 = np.amax(segments[:, 2])
    maxx36 = np.amax(segments[:, 6])
    maxx3 = np.amax(np.array([maxx32, maxx36]))
    
    print("original x_3 minimum =", minx3_old)
    print("original x_3 maximum =", maxx3_old)
    print("new x_3 minimum =", minx3)
    print("new x_3 maximum =", maxx3)
    
    # Setting dimensions of the box
    lft = minx1
    rght = maxx1
    frnt = minx2
    bck = maxx2
    bttm = minx3
    tp = 0
    
    # Saving out domain dimensions
    domain_dimensions = np.array([[lft, rght],
                                  [frnt, bck],
                                  [bttm, tp]])
    
    np.savetxt('data/domain_dimensions_global.txt', domain_dimensions)
    
    # Creating the box mesh
    point0 = Point(lft, frnt, bttm)
    point1 = Point(rght, bck, tp)
    
    mesh = BoxMesh(point0, point1, refinement, refinement, refinement)    
    
    print('Dimensions of domain')
    print('length = ', rght - lft)
    print('width = ', bck - frnt)
    print('depth = ', tp - bttm)
    
    # Base of soil domain
    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return(near(x[2], bttm))
    
    # Soil/atmosphere interface    
    class Top(SubDomain):
        def inside(self, x, on_boundary):
            return(near(x[2], tp))
    
    # Lateral soil/soil boundary 
    class Left(SubDomain):    
        def inside(self, x, on_boundary):
            return(near(x[0], lft))
    
    # Lateral soil/soil boundary 
    class Right(SubDomain):    
        def inside(self, x, on_boundary):
            return(near(x[0], rght)) 
    
    # Lateral soil/soil boundary     
    class Front(SubDomain):
        def inside(self, x, on_boundary):
            return(near(x[1], frnt))
    
    # Lateral soil/soil boundary    
    class Back(SubDomain):
        def inside(self, x, on_boundary):
            return(near(x[1], bck))
        
    # Initialising subdomain instances for box domain.
    bottom = Bottom()
    top = Top()
    left = Left()
    right = Right()
    front = Front()
    back = Back()
    
    # Initialising mesh functions for boundaries of box domain.
    boundaries = MeshFunction("size_t", mesh, 2)
    boundaries.set_all(0)
    
    bottom.mark(boundaries, 1)
    top.mark(boundaries, 2)
    left.mark(boundaries, 3)
    right.mark(boundaries, 4)
    front.mark(boundaries, 5)
    back.mark(boundaries, 6)
    
    # Defining measure.
    dx = Measure('dx', mesh)
    
    # Defining measures corresponding to boundary surfaces.
    ds = Measure('ds', domain = mesh, subdomain_data = boundaries)
    
    # Saving mesh.
    File(f'data/mesh_global_boxmesh.xml.gz') << mesh
    
    return mesh

##############################################################################
# Building a mesh with the mesher_global function
##############################################################################

mesher_global(20)
    