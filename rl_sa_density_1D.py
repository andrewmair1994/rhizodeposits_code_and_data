# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:53:00 2024

@author: andre
"""

# This is a code to generates 1-dimensional density functions by 
# integrating the 3D functions over their lateral 
# dimensions.

###############################################################################
# Importing necessary libaries
###############################################################################

from fenics import *
from dolfin import *
from mshr import *
import numpy as np
import math as math

###############################################################################
# Defining a function that constructs the 1D density functions over a 1D domain
# with a given number of nodes.
###############################################################################

# Input to interval mesh is number of intervals, so number of nodes is this  
# this number plus 1.

def densities_1D(name, Nx, N_int, gamma = 2.0):
    """
    
    Parameters
    ----------
    name : string
        Name of the root system in question.
    Nx : Integer
        Level of refinement of vertical domain.
    N_int : Integer
        Number of sample points for quadrature method.
    gamma : Float, optional
        The initial factor multiplying the entries of the covariance matrix 
        that is used to define the gaussian density functions for each segment 
        of the root system. Default value gamma = 2.

    Returns
    -------        
    RLD_1D : np.array(Nx +1, )
        Nodal values of 1D root length density function
    NRLD_1D : np.array(Nx +1, )
        Nodal values of 1D normalised root length density
        
    RSA_1D : np.array(Nx +1, )
        Nodal values of laterally averaged root surface area density
        
    """
    
    segments = np.loadtxt(f'data/{name}formatted.txt')

    # Split the gamma_v values into integer and decimal components for saving.
    gamma_v_int = str(math.modf(gamma)[1])[:-2]
    gamma_v_dec = str(np.round(math.modf(gamma)[0], 2))[2:]

    # Split the gamma_l values into integer and decimal components for saving
    gamma_l_int = str(math.modf(gamma)[1])[:-2]
    gamma_l_dec = str(np.round(math.modf(gamma)[0], 2))[2:]
    
    # Split the gamma_sa values into integer and decimal components for saving
    gamma_sa_int = str(math.modf(gamma)[1])[:-2]
    gamma_sa_dec = str(np.round(math.modf(gamma)[0], 2))[2:]

    ###########################################################################
    # Defining the soil domain that contains the root system
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

    print("original x3 minimum =", minx3_old)
    print("original x3 maximum =", maxx3_old)
    print("new x3 minimum =", minx3)
    print("new x3 maximum =", maxx3)
    
    # Import global domain dimensions   
    domain_dimensions = np.loadtxt('data/domain_dimensions_global.txt')
        
    # Setting dimensions of the domain
    lft = domain_dimensions[0, 0]
    rght = domain_dimensions[0, 1]
    frnt = domain_dimensions[1, 0]
    bck = domain_dimensions[1, 1]
    bttm = domain_dimensions[2, 0]
    tp = domain_dimensions[2, 1]
        
    print('left boundary', lft)
    print('right boundary', rght)
    print('front boundary', frnt)
    print('back boundary', bck)
    print('bottom boundary', bttm)
    print('tp boundary', tp)    
        
    # Creating a domain 
    domain = Box(Point(lft, frnt, bttm), Point(rght, bck, tp))
    
    # Computing lateral area of box domain.
    Alat = (rght - lft)*(bck - frnt)
    print('Lateral area =', Alat)

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
        
    mesh = Mesh(f'data/mesh_global.xml.gz')
        
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
    
    # Function space for volumetric root density.
    W = FunctionSpace(mesh, 'CG', 1)
    
    ###########################################################################
    # Reading in 3D root length density and root surface area functions.
    ###########################################################################
    
    RLD = Function(W)
    NRLD = Function(W)
    RSA = Function(W)
    
    RLD_store = TimeSeries(f'data/{name}_store_len_gaml{gamma_l_int}_{gamma_l_dec}_global')
    NRLD_store = TimeSeries(f'data/{name}_store_nlen_gaml{gamma_l_int}_{gamma_l_dec}_global')
    RSA_store = TimeSeries(f'data/{name}_store_rsa_gamsa{gamma_sa_int}_{gamma_sa_dec}_global')
        
    RLD_store.retrieve(RLD.vector(), 1)
    NRLD_store.retrieve(NRLD.vector(), 1)
    RSA_store.retrieve(RSA.vector(), 1)
    
    ###########################################################################
    # Creating empty vectors to store the nodal values of the laterally averaged
    # root density profiles.
    ###########################################################################

    RLD_1D = np.zeros(Nx + 1)
    NRLD_1D = np.zeros(Nx + 1) 
    RSA_1D = np.zeros(Nx + 1)
    
    ###########################################################################
    # Calculating laterally averaged volumetric density profiles
    ###########################################################################
    
    # Vector of x3 coordinates at which to compute a lateral average.
    x3s = np.linspace(tp, bttm, Nx + 1)
    
    # x1 samples and weights
    Nx1 = 100
    xis, w1s = np.polynomial.legendre.leggauss(Nx1)

    # x2 samples and weights
    Nx2 = 100
    etas, w2s = np.polynomial.legendre.leggauss(Nx2)
    
    # Function for x1 as a function of xi
    def x_xi(xi):
        return ((rght - lft)*xi + (rght + lft))/2
    print('x1 sample points', x_xi(xis))

    # Function for x2 as a function of eta
    def y_eta(eta):
        return ((bck - frnt)*eta + (bck + frnt))/2
    print('x2 sample points', y_eta(etas))    
    
    for x3 in range(Nx + 1):
        lat_I_RLD = 0
        lat_I_NRLD = 0
        lat_I_RSA = 0
        for i in range(Nx1):
            for j in range(Nx2):
                print('spatial sample point =', (x_xi(xis[i]), y_eta(etas[j]), x3s[x3]))
                
                if x3s[x3] != bttm:
                    lat_I_RLD += (rght - lft)*(bck - frnt)*(1/4)*w1s[i]*w2s[j]*RLD(x_xi(xis[i]), y_eta(etas[j]), x3s[x3])
                    lat_I_NRLD += (rght - lft)*(bck - frnt)*(1/4)*w1s[i]*w2s[j]*NRLD(x_xi(xis[i]), y_eta(etas[j]), x3s[x3])
                    lat_I_RSA += (rght - lft)*(bck - frnt)*(1/4)*w1s[i]*w2s[j]*RSA(x_xi(xis[i]), y_eta(etas[j]), x3s[x3])
                
                else:
                    lat_I_RLD += (rght - lft)*(bck - frnt)*(1/4)*w1s[i]*w2s[j]*RLD(x_xi(xis[i]), y_eta(etas[j]), x3s[x3] + 1E-10)
                    lat_I_NRLD += (rght - lft)*(bck - frnt)*(1/4)*w1s[i]*w2s[j]*NRLD(x_xi(xis[i]), y_eta(etas[j]), x3s[x3] + 1E-10)
                    lat_I_RSA += (rght - lft)*(bck - frnt)*(1/4)*w1s[i]*w2s[j]*RSA(x_xi(xis[i]), y_eta(etas[j]), x3s[x3] + 1E-10)
            
        RLD_1D[x3] = lat_I_RLD/Alat
        NRLD_1D[x3] = lat_I_NRLD/Alat
        RSA_1D[x3] = lat_I_RSA/Alat
    
    # Saving out the 1D nodal values
    
    else:
        np.savetxt(f'data/{name}_{Nx}_RSA1D_gamsa{gamma_sa_int}_{gamma_sa_dec}_node_vals.txt', RSA_1D)
        np.savetxt(f'data/{name}_{Nx}_RLD1D_gaml{gamma_l_int}_{gamma_l_dec}_node_vals.txt', RLD_1D*Alat)
        np.savetxt(f'data/{name}_{Nx}_NRLD1D_gaml{gamma_l_int}_{gamma_l_dec}_node_vals.txt', NRLD_1D*Alat)
        
    return RLD_1D, NRLD_1D, RSA_1D
    

RLD_1D, NRLD_1D, RSA_1D = densities_1D('trigo30days', 100, 20)