# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:54:35 2024

@author: andre
"""

# This is a FEniCS code that constructs functions for root length density, 
# normalised root length density and surface area density from CRootBox
# architecture data. The functions for each root system can be generated
# by putting the appropriate name as the input in the python function
# named "densities".

# Visualisations of the density functions (xdmf files) are saved to the data 
# folder.

###############################################################################
# Importing necessary libaries
###############################################################################

from fenics import *
from dolfin import *
from mshr import *
import numpy as np
from numpy import linalg as la
from scipy.stats import multivariate_normal
import math as math

###############################################################################
# Defining rotation function which is used in the computation of the density 
# functions.
###############################################################################

def rotation(v):
    """
    

    Parameters
    ----------
    v : np.array(3,)
        A chosen vector. In this application v represents the orientation of 
        a root segment, but the function could be applied to a direction 
        vector arising from any context.

    Returns
    -------
    R : np.array(3,3)
        The rotation matrix that rotates the canonical x_3-axis onto v.
    
    R_ : np.array(3,3)
         The inverse of the rotation matrix R, it rewrites a vector in terms
         of the basis by v and it's two perpindicular axes.'.

    """
    
    # Computing norm of root segment direction vector
    v_norm = la.norm(v)
    
    # Computing normalised vector for root direction
    v_hat = v/v_norm
    
    # Canonical x_3-axis
    up = np.array([0, 0, 1])
    
    # Negative x_3-axis
    down = np.array([0, 0, -1])
    
    # Checking that the segment is not already parallel to positive or
    # negative x_3-axis        
    if not (np.all(v_hat == up) or np.all(v_hat == down)):
    
        # Computing cross product
        cp = np.cross(up, v_hat)
        cp_norm = la.norm(cp)
        
        # Normalising cross product for unitary axis of rotation
        y = cp/cp_norm
        
        # Computing angle between canonical x_3-axis and v_hat around this axis of rotation
        up_norm = la.norm(up)
        v_hat_norm = la.norm(v_hat)
        up_dot_v_hat = np.dot(up, v_hat)
        cos_gamma = up_dot_v_hat/(up_norm*v_hat_norm)
        gamma = np.arccos(cos_gamma)
        
        # Computing cross product matrix
        Y = np.array([[0, -y[2], y[1]],
                      [y[2], 0, -y[0]],
                      [-y[1], y[0], 0]])
        
        # Square of cross product matrix
        Y_squared = np.dot(Y,Y)
        
        # Identity matrix
        I = np.identity(3)
        
        # Computing sin(gamma)
        sin = np.sin(gamma)
        
        # Computing cos(gamma)
        cos = np.cos(gamma)
        
        # Computing rodrigues' rotation matrix
        R = I + (sin*Y) + ((1 - cos)*Y_squared)
        
        # Computing rotated vector
        R_ = la.inv(R)
        
    # If the segment is parallel to the negative x_3-axis then rodrigues formula
    # need not be used, R is the  identity matrix with a negative in the final
    # diagonal entry
    elif np.all(v_hat == down):
            
        R = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, -1]])
        R_ = la.inv(R)
        
    # If the segment is parallel to the positive x_3-axis then no rotation is 
    # required and R is the identity matrix
    else:
            
        R = np.identity(3)
        R_ = la.inv(R)
            
    return R, R_


###############################################################################
# Defining a function which generates the nodal values of a root length
# density function and a normalised root length density function
# for given root system data.
###############################################################################

def root_length_density_constructor(segments,
                                    W,
                                    gamma,
                                    dx):
    """
    Parameters
    ----------
    segments : np.array([num_segs, 8])
        An array containing the start and end points of each root 
        segment along with the diameters of each end point of the
        segment.
                
    W : dolfin FunctionSpace
        A scalar function space onto which the density function is
        projected.
        
    gamma : Float > 0
        The scale on the entries of the covariance matrix used to define
        the gaussian root length density functions for each segment of the 
        root system.
        
    dx : dolfin Measure
        The measure associated with the mesh upon which W is defined.
            
    Returns
    -------
    RLD_nodal : np.array([num_scal_nodes,])
        Vector containing the nodal values of the root 
        length density function.
    
    NRLD_nodal : np.array([num_scal_nodes,])
        Vector containing the nodal values of the normalised 
        root length density function.              
    
    integrated_root_length_density : Float
        Integral of the root length density function over the soil domain,
        which we want to equal architecture_length.
                                    
    integrated_normalised_root_length_density : Float
        Integral of the normalised root length density function over the soil
        domain, which we want to equal 1.
    
    architecture_length : Float
        Combined length of roots in the system.                                 
    
    tol_l : FLoat
        Acceptable tolerance for difference between integrated root length
        density and architecture length.          
    """
    
    # Function for root length density.
    RLD = Function(W)
    
    # Function for normalised root length density.
    NRLD = Function(W)
    
    # Array of nodes for scalar functions (shape = N_scalar_nodes x 3).
    scalar_nodes = W.tabulate_dof_coordinates()
    
    # Number of nodes for scalar functions.
    N_scalar_nodes = len(scalar_nodes[:, 0])
    
    # Number of segments in the root system.
    N_segments = len(segments[:, 0])
    
    # Vector containing the length of each segment.
    segment_lengths = np.sqrt((segments[:, 0] - segments[:, 4])**2 
                              + (segments[:, 1] - segments[:, 5])**2  
                              + (segments[:, 2] - segments[:, 6])**2)
    
    # Vector containing the radius of end one of each segment.
    proximal_radii = segments[:, 3]
    
    # Vector containing the radius of end two of each segment. 
    distal_radii = segments[:, 7]
    
    # Combined length of roots within the system.
    architecture_length = np.sum(segment_lengths)
    
    # Starting the scheme 
    # to iteratively construct the length density functions.
    for i in range(N_segments):
        
        # Coordinates of end one of segment i.
        a0 = segments[i, 0]
        a1 = segments[i, 1]
        a2 = segments[i, 2]
        
        # Radius of end one of segment i.
        r_a = proximal_radii[i]
        
        # Coordinates of end two of segment i.
        b0 = segments[i, 4]
        b1 = segments[i, 5]
        b2 = segments[i, 6]
        
        # Direction vector of root: v.
        if a2 < b2:
            v = np.array([a0 - b0, a1 - b1, a2 - b2])
        else:
            v = np.array([b0 - a0, b1 - a1, b2 - a2])
            
        # Radius of end two of segment i.
        r_b = distal_radii[i]
    
        # Average radius.
        r = (r_a + r_b)/2
    
        # Length of the segment.
        segment_length = segment_lengths[i]
    
        # Computing coordinates of the centre of the root segment.
        beta0 = (a0 + b0)/2
        beta1 = (a1 + b1)/2
        beta2 = (a2 + b2)/2

        # Formulating the centre point of the root segment as the mean vector
        # of a multivariate normal distribution.
        beta = np.array([beta0, beta1, beta2])
            
        # Computing the covariance matrix for an unoriented gaussian approximation.
        # The variances in the x1 and x2 directions are proportional to the average 
        # radius of the segment r and the variance in the x3 direction is 
        # proportional to the length of the segment l. The constant of 
        # of proportionality gamma can be tuned to achieve a blurrier or sharper 
        # resolution of the underlying architecture.

        C = np.array([[gamma*r, 0, 0],
                      [0, gamma*r, 0],
                      [0, 0, gamma*0.5*segment_length]])

        # Obtaining both basis change matrices associated to the direction of the 
        # segment.
        R, R_ = rotation(v)

        # Applying the basis change matrices to the covariance matrix to yield
        # the covariance matrix for the gaussian approximation of the segment
        # that is also oriented in the direction of the segment.
        C_tilde = np.dot(R,np.dot(C, R_))

        # Obtaining the nodal values of the multivariate gaussian function which
        # is the pdf of a tri-variate noramll random variable with covariance 
        # C_tilde and mean vector beta.
        f_nodal = multivariate_normal.pdf(scalar_nodes, mean = beta, cov = C_tilde)
        
        # Initialising a function to take the nodal values above.
        f = Function(W)
        
        # Setting the nodal values of the initialised function to the
        # nodal values of the tri-variate gaussian density function.
        f.vector()[:] = f_nodal
        
        # Computing the integral of the function f.
        integrated_f = assemble(f*dx)
        
        # # Printing the integral of the pdf of the multivariate normal for this 
        # # segment.
        # print('Integral of pdf for this segment =', integrated_f)
        
        # Adding the scaled segment density function to the 
        # overall root density function.
        if integrated_f <= 1E-15:    
            RLD.vector()[:] += f.vector()[:]
        else:
            RLD.vector()[:] += (segment_length/integrated_f)*f.vector()[:]
            
    integrated_length_density = assemble(RLD*dx)

    RLD_nodal = RLD.vector()[:]
    NRLD_nodal = (1/integrated_length_density)*RLD.vector()[:]
    NRLD.vector()[:] = NRLD_nodal
    
    integrated_normalised_length_density = assemble(NRLD*dx)
       
    # Tolerance for difference between integrated_length_density and architecture_length
    tol_l = 0.05*architecture_length
    
    return RLD_nodal, NRLD_nodal, integrated_length_density, integrated_normalised_length_density, architecture_length, tol_l

###############################################################################
# Defining a function which generates the nodal values of a root surface area
# density function for given root system data.
###############################################################################

def rsa_density_constructor(segments,
                            W,
                            gamma,
                            dx):
    """
    

    Parameters
    ----------
    segments : np.array
        An array containing the start and end points of each root 
        segment along with the diameters of each end point of the
        segment.
    W : dolfin FunctionSpace
        A scalar function space onto which the density function is
        projected.
    gamma : Float > 0
        The scale on the entries of the covariance matrix used to define
        the gaussian surface area density functions for each segment of the 
        root system.
    dx : dolfin Measure
        The measure associated with the mesh upon which W is defined.

    Returns
    -------
    RSA_nodal : np.array([num_scal_nodes,])
        Vector containing the value of the proposed surface area
        density function on each node of the meshed domain.
    integrated_surface_area_density : Float
        Integral of the surface area density function over the soil domain,
        which we want to equal architecture_lateral_area.
    architecture_lateral_area : Float
        Lateral surface area of root system.
    tol_sa : FLoat
        Acceptable tolerance for difference between integrated surface area
        density and architecture surface area. 

    """
    
    # Function for proposed surface area density.
    RSA = Function(W)
    
    # Array of nodes for scalar functions (shape = N_scalar_nodes x 3).
    scalar_nodes = W.tabulate_dof_coordinates()
    
    # Number of nodes for scalar functions.
    N_scalar_nodes = len(scalar_nodes[:, 0])
    
    # Defining pi
    pi = np.pi
    
    # Number of segments in the root system
    N_segments = len(segments[:, 0])
    
    # Vector containing the length of each segment.
    segment_lengths = np.sqrt((segments[:, 0] - segments[:, 4])**2 
                             + (segments[:, 1] - segments[:, 5])**2  
                             + (segments[:, 2] - segments[:, 6])**2)
    
    # Vector containing the radius of end one of each segment.
    proximal_radii = segments[:, 3]
    
    # Vector containing the radius of end two of each segment. 
    distal_radii = segments[:, 7]
    
    # Vector containing the lateral surface area of each segment.
    segment_lateral_areas = pi*(proximal_radii + distal_radii)*np.sqrt(segment_lengths**2 + (proximal_radii - distal_radii)**2)
    
    # surface ares of the root system within the soil column.
    architecture_surface_area = np.sum(segment_lateral_areas)
    
    # Starting the scheme to iteratively construct the density function and
    # flow anisotropy matrix.
    for i in range(N_segments):
        
        # Coordinates of end one of segment i.
        a0 = segments[i, 0]
        a1 = segments[i, 1]
        a2 = segments[i, 2]
        
        # Radius of end one of segment i.
        r_a = proximal_radii[i]
        
        # Coordinates of end two of segment i.
        b0 = segments[i, 4]
        b1 = segments[i, 5]
        b2 = segments[i, 6]
        
        # Direction vector of root: v.
        if a2 < b2:
            v = np.array([a0 - b0, a1 - b1, a2 - b2])
        else:
            v = np.array([b0 - a0, b1 - a1, b2 - a2])
            
        # Radius of end two of segment i.
        r_b = distal_radii[i]
    
        # Average radius.
        r = (r_a + r_b)/2
    
        # Length of the segment.
        l = segment_lengths[i]
    
        # Lateral area of segment (volume of a conical frustum).
        segment_lateral_area = segment_lateral_areas[i]
        
        # Computing coordinates of the centre of the root segment.
        beta0 = (a0 + b0)/2
        beta1 = (a1 + b1)/2
        beta2 = (a2 + b2)/2

        # Formulating the centre point of the root segment as the mean vector
        # of a multivariate normal distribution.
        beta = np.array([beta0, beta1, beta2])
            
        # Computing the covariance matrix for an unoriented gaussian approximation.
        # The variances in the x1 and x2 directions are proportional to the average 
        # radius of the segment r and the variance in the x3 direction is 
        # proportional to the length of the segment l. The constant of 
        # of proportionality gamma can be tuned to achieve a blurrier or sharper 
        # resolution of the underlying architecture.

        C = np.array([[gamma*r, 0, 0],
                      [0, gamma*r, 0],
                      [0, 0, gamma*0.5*l]])

        # Obtaining both basis change matrices associated to the direction of the 
        # segment.
        R, R_ = rotation(v)

        # Applying the basis change matrices to the covariance matrix to yield
        # the covariance matrix for the gaussian approximation of the segment
        # that is also oriented in the direction of the segment.
        C_tilde = np.dot(R,np.dot(C, R_))

        # Obtaining the nodal values of the multivariate gaussian function which
        # is the pdf of a tri-variate noramll random variable with covariance 
        # C_tilde and mean vector beta.
        f_nodal = multivariate_normal.pdf(scalar_nodes, mean = beta, cov = C_tilde)
        
        # Initialising a function to take the nodal values above.
        f = Function(W)
        
        # Setting the nodal values of the initialised function to the
        # nodal values of the tri-variate gaussian density function.
        f.vector()[:] = f_nodal
        
        # Computing the integral of the function f.
        integrated_f = assemble(f*dx)
        
        # Adding the scaled segment density function to the 
        # overall root density function.
        if integrated_f <= 1E-15:    
            RSA.vector()[:] += f.vector()[:]
        else:
            RSA.vector()[:] += (segment_lateral_area/integrated_f)*f.vector()[:]
            
    # Computing the integral of the density functions over the domain.
    integrated_surface_area_density = assemble(RSA*dx)

    RSA_nodal = RSA.vector()[:]
        
    # Tolerance for difference between integrated_density and architecture_area
    tol_sa = 0.05*architecture_surface_area
    
    return RSA_nodal, integrated_surface_area_density,  architecture_surface_area, tol_sa

###############################################################################
# Defining a function which constructs the root length density functions and a 
# surface area density function.
###############################################################################

def densities(name,
              gamma = 2.0):
    
    """
    Parameters
    ----------
    name : string
        Name of the root system
    
    gamma : Float > 0
        The initial factor multiplying the entries of the covariance matrix 
        that is used to define the gaussian density functions for each segment 
        of the root system. Default value gamma = 2.
        
    Returns
    -------
        
    RLD_nodal : np.array([num_scal_nodes,])
        Vector containing the nodal values of the root 
        length density function.
    
    NRLD_nodal : np.array([num_scal_nodes,])
        Vector containing the nodal values of the normalised 
        root length density function.
    RSA_nodal : np.array([num_scal_nodes,])
        Vector containing the nodal values of the normalised root surface area
        densitwy function          

    """
    
    ###########################################################################
    # Importing crootbox root system data for this plant and root system combo.
    ###########################################################################
    
    segments = np.loadtxt(f'data/{name}formatted.txt')
    
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
    
    # Import global domain dimensions   
    domain_dimensions = np.loadtxt('data/domain_dimensions_global.txt')
        
    # Setting dimensions of the domain
    lft = domain_dimensions[0, 0]
    rght = domain_dimensions[0, 1]
    frnt = domain_dimensions[1, 0]
    bck = domain_dimensions[1, 1]
    bttm = domain_dimensions[2, 0]
    tp = domain_dimensions[2, 1]
        
    # Creating a domain 
    domain = Box(Point(lft, frnt, bttm), Point(rght, bck, tp))
    
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
    # Constructing the root length density function and normalised root length
    # density function for the root system.
    ###########################################################################
    
    # Initiating a function for the root length density.
    RLD = Function(W)

    # Initiating a function for the normalised root length density.
    NRLD = Function(W)
    
    # Generating nodal values of root length density.
    RLD_nodal, NRLD_nodal, integrated_length_density, integrated_normalised_length_density, architecture_length, tol_l = root_length_density_constructor(segments, W, gamma, dx) 
    print('densities: Absolute difference between integrated root length density function and architecture length at gamma = ', gamma, 'is',  abs(architecture_length - integrated_length_density))
    print('densities: Tolerance for absolute difference between integrated root length density function and architecture length is',  tol_l)
    
    # If the absolute difference between the architecture length and the 
    # integrated root length density is greater than the prescribed tolerance tol_l,
    # then gamma is increased until this is no longer the case.
    while abs(architecture_length - integrated_length_density) > tol_l:
        
        print('densities: Absolute difference between architecture length and integrated length density is greater than tolerance so recomputing with larger gamma')
        # Increasing gamma
        gamma *= (1.025)    
        new_RLD_nodal, new_NRLD_nodal, new_integrated_length_density, new_integrated_normalised_length_density, architecture_length, tol_l = root_length_density_constructor(segments, W, gamma, dx) 
        print('densities: Tolerance for absolute difference between integrated length density and architecture length =', tol_l)
        print('densities: Absolute difference between new integrated length density and architecture length =', abs(architecture_length - new_integrated_length_density))
            
        # If absolute difference between new integrated density and architecture
        # volume is greater than before then loop is broken
        if abs(architecture_length - integrated_length_density) - abs(architecture_length - new_integrated_length_density) < 0:
            print('densities: Cannot achieve desired tolerance for difference between integrated length density and architecture length')
            break
        
        else:
            print('densities: Absolute difference between new integrated length density and architecture length is less than before, so new density is accepted')
            RLD_nodal = new_RLD_nodal
            NRLD_nodal = new_NRLD_nodal
            integrated_length_density = new_integrated_length_density
            
    # Defining the functions for root length density
    # rho_n and norm_rho_n are defined.
    RLD.vector()[:] = RLD_nodal
    NRLD.vector()[:] = NRLD_nodal
    
    print('densities: Maximum value of final root length density =', np.max(RLD.vector()[:]))
    print('densities: Maximum value of final normalised root length density =', np.max(NRLD.vector()[:]))
    print('densities: Tolerance for absolute difference between integrated length density and architecture length =', tol_l)
    print('densities: Absolute difference between new integrated length density and architecture length =', abs(architecture_length - integrated_length_density))
    
    # Split the gamma_l values into integer and decimal components for saving
    gamma_l_int = str(math.modf(gamma)[1])[:-2]
    gamma_l_dec = str(np.round(math.modf(gamma)[0], 2))[2:]
    print("Integer component of gamma_l =", gamma_l_int)
    print("Decimal component of gamma_l =", gamma_l_dec)
    
    np.savetxt(f'data/{name}_gaml{gamma_l_int}_{gamma_l_dec}_global.txt', np.array([gamma]))
        
    ###########################################################################
    # Constructing the surface area density function for the root system.
    ###########################################################################
    
    # Function for proposed surface area density.
    RSA = Function(W)
    
    # Generating nodal values for surface area density and heterogeneity matrix
    RSA_nodal, integrated_surface_area_density, architecture_surface_area, tol_sa = rsa_density_constructor(segments, W, gamma, dx)
    print('densities: Maximum value of surface area density function at gamma = ', gamma, 'is',  np.max(RSA_nodal))
    print('densities: Tolerance for absolute difference between integral of surface area density function and architecture surface area is tol_sa =', tol_sa)
    print('densities: Absolute difference between integral of surface area density function and architecture surface area =', abs(architecture_surface_area - integrated_surface_area_density))
    
    # If the absolute difference between the architecture length and the 
    # integrated surface area density is greater than the prescribed tolerance tol_sa,
    # then gamma is increased until this is no longer the case.
    while abs(architecture_surface_area - integrated_surface_area_density) > tol_sa:
        
        print('densities: Absolute difference between architecture surface area and integrated surface area density is greater than tolerance so recomputing with larger gamma')
        # Increasing gamma
        gamma *= (1.025)    
        new_RSA_nodal, new_integrated_surface_area_density, architecture_surface_area, tol_sa = rsa_density_constructor(segments, W, gamma, dx) 
        print('densities: Tolerance for absolute difference between integrated surface area density and architecture surface area  =', tol_sa)
        print('densities: Absolute difference between new integrated surface area density and architecture surface area  =', abs(architecture_surface_area - new_integrated_surface_area_density))
            
        # If absolute difference between new integrated density and architecture
        # surface area is greater than before then loop is broken
        if abs(architecture_surface_area - integrated_surface_area_density) - abs(architecture_surface_area - new_integrated_surface_area_density) < 0:
            print('densities: Cannot achieve desired tolerance for difference between integrated surface area density and architecture surface area ')
            break
        
        else:
            print('densities: Absolute difference between new integrated surface area density and architecture surface area  is less than before, so new density is accepted')
            RSA_nodal = new_RSA_nodal
            integrated_surface_area_density = new_integrated_surface_area_density
                
    # Once the surface area density function is below 1 and the tolerance between
    # the integral of the surface area density and the architecture surface area is 
    # satisfied, then RSA is defined.
    RSA.vector()[:] = RSA_nodal
    
    print('densities: Maximum value of final surface area density =', np.max(RSA.vector()[:]))
    print('densities: Tolerance for absolute difference between integrated surface area density and architecture surface area =', tol_sa)
    print('densities: Absolute difference between final integrated surface area density and architecture surface area =', abs(architecture_surface_area - integrated_surface_area_density))

    # Split the gamma_sa values into integer and decimal components for saving.
    gamma_sa_int = str(math.modf(gamma)[1])[:-2]
    gamma_sa_dec = str(np.round(math.modf(gamma)[0], 2))[2:]
        
    np.savetxt(f'data/{name}_gamsa{gamma_sa_int}_{gamma_sa_dec}_global.txt', np.array([gamma]))
        
    ###########################################################################
    # Post processing of density functions
    ###########################################################################
    
    RLD_store = TimeSeries(f'data/{name}_store_len_gaml{gamma_l_int}_{gamma_l_dec}_global')
    NRLD_store = TimeSeries(f'data/{name}_store_nlen_gaml{gamma_l_int}_{gamma_l_dec}_global')
    RSA_store = TimeSeries(f'data/{name}_store_rsa_gamsa{gamma_sa_int}_{gamma_sa_dec}_global')
    
    # Assigning the appropriate values to the stored data.
    RLD_store.store(RLD.vector(), 1)
    NRLD_store.store(NRLD.vector(), 1)
    RSA_store.store(RSA.vector(), 1)
    
    ###########################################################################
    # Saving the visualisations of the density functions
    ###########################################################################
    
    xdmffile_RLD = XDMFFile(f'data/{name}_plot_len_gaml{gamma_l_int}_{gamma_l_dec}_global.xdmf')
    xdmffile_NRLD = XDMFFile(f'data/{name}_plot_nlen_gaml{gamma_l_int}_{gamma_l_dec}_global.xdmf')
    xdmffile_RSA = XDMFFile(f'data/{name}_plot_rsa_gamsa{gamma_sa_int}_{gamma_sa_dec}_global.xdmf')
    
    xdmffile_RLD.write(RLD)
    xdmffile_NRLD.write(NRLD)
    xdmffile_RSA.write(RSA)

    return NRLD_nodal, RSA_nodal

###############################################################################
# Constructing density functions
###############################################################################

NRLD_nodal, RSA_nodal = densities('trigo30days')  
