# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 17:35:27 2025

@author: andre
"""

# This is a code which identifies the depths at which each root system becomes
# most dense. To identify what should be considered as the start of
# the uptake zone.

###############################################################################
# Importing necessary libraries.
###############################################################################

from fenics import *
from dolfin import *
import numpy as np
import math as math

###############################################################################
# Defining function to return depth ranges of uptake zone
###############################################################################

def uptake_zones(Nx, start_frac, gamma = 2.0, imported_mesh = 'def'):
    """
    

    Parameters
    ----------
    Nx : Integer
        Level of spatial refinement.
    start_frac : 0 < Float < 1 
        Fraction of maximum NRLD value above which rooted zone is assumed to
        begin (and below which it is assumed to end)
    gamma : Float, optional
        Width of Gaussian support in 3d root length and surface area density
        functions. The default is 2.0.
    imported_mesh : String
        3D mesh that is imported for the construction of root density functions.
        'def' imports the mesh that was constructed using the deprecated
        functionality mshr, 'box' imports the mesh that can still be
        created using available Legacy FEniCS docker images.    

    Returns
    -------
    start_points : np.array(3,) 
        Vector containing the depths at which uptake zone starts for 
        each root system. First entry for the 6 day old system, second entry 
        for the 15 day old system and third entry for the 30 day old system.
    """
    
    # Split the gamma_l values into integer and decimal components for saving
    gamma_l_int = str(math.modf(gamma)[1])[:-2]
    gamma_l_dec = str(np.round(math.modf(gamma)[0], 2))[2:]
    
    # Split the start_frac into integer and decimal components for saving
    start_frac_int = str(math.modf(start_frac)[1])[:-2]
    start_frac_dec = str(np.round(math.modf(start_frac)[0], 2))[2:]
    
    ###########################################################################
    # Defining the soil domain that contains the root system
    ###########################################################################
    
    # Import global domain dimensions   
    domain_dimensions = np.loadtxt('data/domain_dimensions_global.txt')
    
    # Setting dimensions of the domain
    bttm = domain_dimensions[2, 0]
    tp = domain_dimensions[2, 1]
        
    # Mesh.
    mesh = IntervalMesh(Nx, bttm, tp)
    
    # Base of soil domain
    class Base(SubDomain):
        def inside(self, x, on_boundary):
            return (near(x[0], bttm))
        
    class Top(SubDomain):
        def inside(self, x, on_boundary):
            return (near(x[0], tp))
        
    # Initialising subdomain instances for interval domain.
    base = Base()
    top = Top()
    
    # Initialising mesh functions for boundaries of inerval
    boundaries = MeshFunction("size_t", mesh, 0)
    boundaries.set_all(0)
    
    base.mark(boundaries, 1)
    top.mark(boundaries, 2)
    
    # Defining measure.
    dx = Measure('dx', mesh)
    
    # Defining measures corresponding to domain endpoints.
    ds = Measure('ds', domain = mesh, subdomain_data = boundaries)
    
    # Function space for scalar functions.
    W = FunctionSpace(mesh, 'CG', 1)
    
    # Array of nodes for scalar functions
    scalar_nodes = W.tabulate_dof_coordinates()
    
    # Array of vertices of mesh
    vertices = mesh.coordinates()
    
    ###########################################################################
    # Importing normalised root length density profiles
    ###########################################################################
    
    NRLD_nodal_6days = np.loadtxt(f'data/trigo6days_{Nx}_NRLD1D_gaml{gamma_l_int}_{gamma_l_dec}_node_vals_{imported_mesh}.txt')
    NRLD_6days = Function(W)
    NRLD_6days.vector()[:] = NRLD_nodal_6days
    
    NRLD_nodal_15days = np.loadtxt(f'data/trigo15days_{Nx}_NRLD1D_gaml{gamma_l_int}_{gamma_l_dec}_node_vals_{imported_mesh}.txt')
    NRLD_15days = Function(W)
    NRLD_15days.vector()[:] = NRLD_nodal_15days
    
    NRLD_nodal_30days = np.loadtxt(f'data/trigo30days_{Nx}_NRLD1D_gaml{gamma_l_int}_{gamma_l_dec}_node_vals_{imported_mesh}.txt')
    NRLD_30days = Function(W)
    NRLD_30days.vector()[:] = NRLD_nodal_30days
    
    # Initialising np.array containing start point of uptake zone for each root 
    # system
    start_points = np.zeros(3)
    
    # Initialising np.array containing end point of uptake zone for each root 
    # system
    end_points = np.zeros(3)
    
    # Identifying start point of uptake zone, 6day old root system.
    MaxNRLD6days = max(NRLD_nodal_6days)
    for i in range(len(NRLD_nodal_6days)):
        if (NRLD_nodal_6days[i + 1] >= start_frac*MaxNRLD6days
            and NRLD_nodal_6days[i] < start_frac*MaxNRLD6days):
            start_points[0] = scalar_nodes[i + 1]
            break
            
    # Identifying start point of uptake zone, 15day old root system.
    MaxNRLD15days = max(NRLD_nodal_15days)
    for i in range(len(NRLD_nodal_15days)):
        if (NRLD_nodal_15days[i + 1] >= start_frac*MaxNRLD15days
            and NRLD_nodal_15days[i] < start_frac*MaxNRLD15days):
            start_points[1] = scalar_nodes[i + 1]
            break
            
    # Identifying start point of uptake zone, 30day old root system.
    MaxNRLD30days = max(NRLD_nodal_30days)
    for i in range(len(NRLD_nodal_30days)):
        if (NRLD_nodal_30days[i + 1] >= start_frac*MaxNRLD30days
            and NRLD_nodal_30days[i] < start_frac*MaxNRLD30days):
            start_points[2] = scalar_nodes[i + 1]
            break
        
    # Identifying end point of uptake zone, 6day old root system.
    MaxNRLD6days = max(NRLD_nodal_6days)
    for i in range(len(NRLD_nodal_6days)):
        if (NRLD_nodal_6days[i + 1] < start_frac*MaxNRLD6days
            and NRLD_nodal_6days[i] >= start_frac*MaxNRLD6days):
            end_points[0] = scalar_nodes[i + 1]
            break
            
    # Identifying end point of uptake zone, 15day old root system.
    MaxNRLD15days = max(NRLD_nodal_15days)
    for i in range(len(NRLD_nodal_15days)):
        if (NRLD_nodal_15days[i + 1] < start_frac*MaxNRLD15days
            and NRLD_nodal_15days[i] >= start_frac*MaxNRLD15days):
            end_points[1] = scalar_nodes[i + 1]
            break
            
    # Identifying end point of uptake zone, 30day old root system.
    MaxNRLD30days = max(NRLD_nodal_30days)
    for i in range(len(NRLD_nodal_30days)):
        if (NRLD_nodal_30days[i + 1] < start_frac*MaxNRLD30days
            and NRLD_nodal_30days[i] >= start_frac*MaxNRLD30days):
            end_points[2] = scalar_nodes[i + 1]
            break
    
    print('Scalar nodes =', scalar_nodes)
    print('Max NRLD 6 day old system =', MaxNRLD6days)
    print('Nodal values of 6 day old system =', NRLD_6days.vector()[:])
    print('Start point of uptake zone of 6 day plant', start_points[0])
    print('End point of uptake zone of 6 day plant', end_points[0])
    
    print('Scalar nodes =', scalar_nodes)
    print('Max NRLD 15 day old system =', MaxNRLD15days)
    print('Nodal values of 15 day old system =', NRLD_15days.vector()[:])
    print('Start point of uptake zone of 15 day plant', start_points[1])
    print('End point of uptake zone of 15 day plant', end_points[1])
    
    print('Scalar nodes =', scalar_nodes)
    print('Max NRLD 30 day old system =', MaxNRLD30days)
    print('Nodal values of 30 day old system =', NRLD_30days.vector()[:])
    print('Start point of uptake zone of 30 day plant', start_points[2])
    print('End point of uptake zone of 30 day plant', end_points[2])
    
    # Saving out start_points
    np.savetxt(f'data/start_points_start_frac{start_frac_int}_{start_frac_dec}_Nx{Nx}_NRLD1D_gaml{gamma_l_int}_{gamma_l_dec}_mesh_{imported_mesh}.txt', start_points)
    np.savetxt(f'data/end_points_start_frac{start_frac_int}_{start_frac_dec}_Nx{Nx}_NRLD1D_gaml{gamma_l_int}_{gamma_l_dec}_mesh_{imported_mesh}.txt', end_points)
    
    return start_points, end_points

start_points, end_points = uptake_zones(100, 0.1)

