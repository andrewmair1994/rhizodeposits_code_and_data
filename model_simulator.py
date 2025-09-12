# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 09:17:25 2024

@author: andre
"""

# This is a code which simulates, in 1D, the effect of rhizodeposits on
# the uptake of water by a wheat root system.

###############################################################################
# Importing necessary libraries.
###############################################################################

from fenics import *
from dolfin import *
import numpy as np
import math as math

###############################################################################
# Defining function for surface tension against rhizodeposit concentration.
###############################################################################

def rhiz_surf_tension(x, a, b):
    """
    
    Parameters
    ----------
    x : Float
        Suspended rhizodeposit concentration.
    a : Float
        Multiplier of power law.
    b : Float
        Shift in power law.
    
    Returns
    -------
    gamma: Float
        Surface tension of mucilage solution

    """
    gamma = 47.5 + (72.86 - 47.5)/(1 + np.exp(a*(x-b)))
    
    return gamma

###############################################################################
# Defining function type to be fitted for contact angle against 
# rhizodeposit concentration.
###############################################################################

def rhiz_contact_angle(x, a, b):
    """
    

    Parameters
    ----------
    x : Float
        Dried rhizodeposit concentration
    a : Float
        Exponent of e in the fitted curve.
    b : Float
        Exponent of (1 + e^{-a*x}) in fitting curve.

    Returns
    -------
    omega: Float
            Contact angle of pore surface with rhizodeposits dried on.

    """
    omega = (0)*pi + (3.529/18.0)*pi*(1 + np.exp(-a*x))**(-b)
    
    return omega

###############################################################################
# Defining function for viscosity against rhizodeposit concentration.
###############################################################################

def rhiz_viscosity(x, a, b):
    """
    

    Parameters
    ----------
    x : Float
        Concentration of rhizodeposits [mg cm^{-3}].
    a : Float
        Multiplying factor in exponent of e in the fitted curve.
    b : Float
        Shift in exponent of e in the fitted curve.

    Returns
    -------
    mu: Float
        Viscosity of the mucilage/water solution.

    """
    mu = 0.00089 + (0.00098 - 0.00089)*(1 + np.exp(-a*(x-b)))**(-10)
    
    return mu

###############################################################################
# Defining Van Genuchten functions for hysteretic water content.
###############################################################################

# Defining a function for indicating the wet dry status at a given node
# of the discretised spatial domain.
def nodal_wet_dry_status (h, h_, status_, alpha_vg_d, alpha_vg_w, tol):
    """
    Function for indicating whether we are in a wetting or drying regime at
    a given node.

    Parameters
    ----------
    h : Float
        Pressure head.
    h_ : Float
        Pressure head from previous step.
    status_ : Float
        Status from previous timestep.
    alpha_vg_d : Float
        Drying inverse air entry pressure head
    alpha_vg_w : Float
        Wetting inverse air entry pressure head
    tol : Float, optional
        Tolerance above which the magnitude of a reversal in pressure head 
        evolution direction constitutes an actual reversal.

    Returns
    -------
    status: Float
        The base shape parameter value corresponding to the wetting or drying
        status of the soil at the node in question.

    """
    # Situation where a drying curve changes to a wetting curve.
    if h-h_ > tol and status_ == alpha_vg_d:
        status = alpha_vg_w
        
    # Situation where a wetting curve is changing to a drying curve.
    elif h-h_ < -tol and status_ == alpha_vg_w:
        status = alpha_vg_d
        
    # All other situations where either the magnitude of pressure change is 
    # to small to constitute a reversal in wetting/drying or the direction,
    # i.e. wetting or drying, is the same as the previous time step    
    else:
        status = status_

    return status

# Vectorising pointwise wetting/drying status function.
wet_dry_status = np.vectorize(nodal_wet_dry_status)

# Function for the nodal alpha shape parameters at a given point depending on 
# whether we are in a wetting or drying regime, rhizodeposit concentration levels
# and how saturated the soil is.
def nodal_alpha_vg (status, h, alpha_vg_d, alpha_vg_w, a_st, b_st, a_ca, b_ca, cw, cd):
    """
    

    Parameters
    ----------
    status : Float
        Wetting or drying status at the current time.
    h : Float
        Pressure head.
    alpha_vg_d : Float
        Drying inverse air entry pressure head.
    alpha_vg_w : Float
        Wetting inverse air entry pressure head.
    a_st : Float
        First parameter in surface tension as function of suspended rhizodeposit 
        concentration.
    b_st : Float
        Second parameter in surface tension as function of suspended rhizodeposit 
        concentration.
    a_ca : Float
        First parameter in contact angle as function of dried rhizodeposit 
        concentration.
    b_ca : Float
        Second parameter in surface tension as function of suspended rhizodeposit 
        concentration.
    cw : Float
        Suspended rhizodeposit contentration.
    cd : FLoat
        Dried rhizodeposit concentration.

    Returns
    -------
    alpha_vg : Float
        Inverse of air entry pressure head.

    """
    
    
    # Surface tension of water.
    gamma0 = rhiz_surf_tension(0, a_st, b_st)
    
    # Contact angle of water.
    omega_w0 = rhiz_contact_angle(0, a_ca, b_ca)
    
    # Surface tension of rhizodeposit solution
    gamma = rhiz_surf_tension(cw, a_st, b_st)
    
    # Contact angle of solution with dried rhizodeposit on surface
    omega_w = rhiz_contact_angle(cd, a_ca, b_ca) 
    
    # If the pressure head at the current time step is already zero
    # then we set alpha to zero in order to maintain the water content at 
    # saturation.
    if h >= 0.0:
        alpha_vg = 0.0
        
    # If the soil is on a wetting trajectory.
    elif status == alpha_vg_w:    
            
        # The wetting alpha is assigned.
        alpha_vg = alpha_vg_w*(gamma/gamma0)*(cos(omega_w0)/cos(omega_w))
        
    # The soil must be on a drying trajectory.
    else:
        # The drying alpha is assigned.
        alpha_vg = alpha_vg_d*(gamma0/gamma)*(cos(omega_w)/cos(omega_w0))
        
    return alpha_vg

# Vectorising function for alpha_vg.
vec_alpha_vg = np.vectorize(nodal_alpha_vg)

# Defining a Python function to return a dolfin function of the alpha_vg
# inverse air entry pressure head parameter.
def alpha_vg_func(status, h, alpha_vg_d, alpha_vg_w, a_st, b_st, a_ca, b_ca, cw, cd, W):
    """
    

    Parameters
    ----------
    status : np.array(N_scalar_nodes,)
        Wetting and drying status at all nodes.
        
    h : Dolfin function of space:
        Pressure head at current step.
    
    alpha_vg_d : Float
        Drying inverse air entry pressure head.
    
    alpha_vg_w : Float
        Wetting inverse air entry pressure head.
    
    a_st : Float
        First parameter in surface tension as function of suspended rhizodeposit 
        concentration.
    
    b_st : Float
        Second parameter in surface tension as function of suspended rhizodeposit 
        concentration.
        
    a_ca : Float
        First parameter in contact angle as function of dried rhizodeposit 
        concentration.
    
    b_ca : Float
        Second parameter in contact angle as function of dried rhizodeposit 
        concentration.     
    
    cw : Dolfin function of space
        Suspended rhizodeposit contentration.
        
    cd : Dolfin function of space
        Dried rhizodeposit contentration.
    
    W: Dolfin function space
        Function space onto which the values of the air entry pressure head
        are projected.

    Returns
    -------
    alpha_vg :  Dolfin function
        Function for the inverse air entry pressure head.

    """
    
    alpha_vg = Function(W)
    
    alpha_vg.vector()[:] = vec_alpha_vg(status, h.vector()[:], alpha_vg_d, alpha_vg_w, a_st, b_st, a_ca, b_ca, cw.vector()[:], cd.vector()[:])
    
    return alpha_vg

# Function for effective saturation.
# Effective saturation function.
def se(h, alpha_vg, n_vg, m_vg):
    """
    Function for effective saturation of soil.

    Parameters
    ----------
    h : Float
        Pressure head.
    alpha_vg : Float
        Inverse of the air entry pressure head.
    n_vg : Float
        Shape parameter of Van Genuchten function.    
    m_vg : Float
        1 - 1/n_vg    

    Returns
    -------
    effective saturation.

    """
    return 1/(pow(1 + pow(abs(alpha_vg*h), n_vg), m_vg))

# Function for nodal residual soil water content in hysteretic function 
# for soil water content.
def nodal_theta_r(status, h, h_, theta_, alpha_vg, alpha_vg_w, theta_r0, theta_s0, n_vg, m_vg):
    """
    Function for the residual water content parameter in the hysteretic water 
    content function.

    Parameters
    ----------
    status : Float
        Indicator of the current wetting/drying status of the soil. 
        Takes either alpha_vg_w or alpha_vg_d.
    h : Float
        Pressure head at current step.
    h_ : Float
        Pressure head from previous step.
    theta_ : Float
        Water content value from previous step.     
    alpha_vg : Float
        Inverse of air entry pressure head.
    alpha_vg_w : Float
        Base value for inverse of air entry pressure head when wetting.
    theta_r0 : Float
        Residual water content.
    theta_s0 : Float
        Saturated water content.
    n_vg : Float
        Van Genuchten shape parameter.
    m_vg : Float
        Van Genuchten shape parameter.
    Returns
    -------
    theta_r : FLoat
        The residual water content to be used in the hysteretic water content
        function at the current timestep.
    """
    
    # If pressure head at current linearisation step or previous timestep is 
    # non-negative then theta_r reverts to non-hysteretic value.
    if h >= 0.0 or h_ >= 0.0:
        theta_r = theta_r0
        
    # We are on a wetting trajectory so the residual water content depends upon
    # alpha_vg which may depend upon water content and hence change with time.
    elif status == alpha_vg_w: 

        theta_r = (theta_- theta_s0*se(h_, alpha_vg, n_vg, m_vg))/(1-se(h_, alpha_vg, n_vg, m_vg))
        
    # We are in a drying trajectory or the soil is saturated so the residual 
    # water content is the base value.
    else:
        theta_r = theta_r0
        
    return theta_r

# Vectorising nodal function for theta_r 
vec_theta_r = np.vectorize(nodal_theta_r)

# Defining a Python function to return a dolfin function of the hysteretic
# residual water content function.
def theta_r_func(status, h, h_, theta_, alpha_vg, alpha_vg_w, theta_r0, theta_s0, n_vg, m_vg, W):
    """
    

    Parameters
    ----------
    status : np.array(N_scalar_nodes, )
        Vector indicatinng the wetting or drying status of each node.
    h : Dolfin function
        Pressure head from current step.
    h_ : Dolfin function
        Pressure head from previous step.
    theta_ : Dolfin function
        Water content from previous step.
    alpha_vg : Dolfin function
        Inverse of air entry pressure head.
    alpha_vg_w : Float
        Base value for inverse of air entry pressure head when wetting.
    theta_r0 : Float
        Residual water content.
    theta_s0 : Float
        Saturated water content.
    n_vg : Float
        Van Genuchten shape parameter.
    m_vg : Float
        Van Genuchten shape parameter.
    W : Dolfin function space
        Function space onto which the residual water content value is to be
        projected.

    Returns
    -------
    theta_r : Dolfin function
        Function for residual water content

    """

    theta_r = Function(W)

    theta_r.vector()[:] = vec_theta_r(status, h.vector()[:], h_.vector()[:], theta_.vector()[:], alpha_vg.vector()[:], alpha_vg_w, theta_r0, theta_s0, n_vg, m_vg)

    return theta_r

# Function for nodal saturated soil water content in hysteretic function 
# for soil water content.
def nodal_theta_s(status, h, h_, theta_, alpha_vg, alpha_vg_d, theta_r0, theta_s0, n_vg, m_vg):
    """
    Function for the saturated water content parameter in the hysteretic water 
    content function.

    Parameters
    ----------
    status : Float
        Indicator of the current wetting/drying status of the soil. 
        Takes either alpha_vg_w or alpha_vg_d.
    h : Float
        Pressure head from current step.
    h_ : Float
        Pressure head from previous step.
    theta_ : Float
        Water content value from previous step.
    alpha_vg : Float
        Inverse of air entry pressure head.
    alpha_vg_d : Float
        Base value for inverse of air entry pressure head when drying.
    theta_r0 : Float
        Residual water content.    
    theta_s0 : Float    
        Saturated water content.
    n_vg : Float
        Van Genuchten shape parameter.
    m_vg : Float
        Van Genuchten shape parameter.
    Returns
    -------
    theta_s : Float
        The saturated water content to be used in the hysteretic water content
        function at the current time step.
    """
    
    # If pressure head at current or previous step
    # is non-negative then saturated water content takes its base value.
    if h >= 0.0 or h_ >= 0.0:
        theta_s = theta_s0
    
    # We are in a drying curve so saturated water content changes with alpha_vg 
    # which depends upon volumetric root density.
    elif status == alpha_vg_d:
            
        theta_s = (theta_ - theta_r0*(1 - se(h_, alpha_vg, n_vg, m_vg)))/se(h_, alpha_vg, n_vg, m_vg)
    
    # We are in a wetting trajectory or the soil is saturated so saturated 
    # water content takes its base value.
    else:
        theta_s = theta_s0
        
    return theta_s

# Vectorising the function for saturated water content.
vec_theta_s = np.vectorize(nodal_theta_s)

# Defining a python function to return a dolfin function of the saturated
# hysteretic water content function.
def theta_s_func(status, h, h_, theta_, alpha_vg, alpha_vg_d, theta_r0, theta_s0, n_vg, m_vg, W):
    """
    

    Parameters
    ----------
    status : np.array(N_scalar_nodes, )
        Vector stating the wetting and drying status at each node.
    h : Dolfin function
        Pressure head at current step.
    h_ : Dolfin function
        Pressure head at previous step.
    theta_ : Dolfin function
        Water content at previous step.
    alpha_vg : Dolfin function
        Inverse of air entry pressure head.
    alpha_vg_d : Float
        Base value for inverse of air entry pressure head when drying.
    theta_r0 : Float
        Residual water content.    
    theta_s0 : Float    
        Saturated water content.
    n_vg : Float
        Van Genuchten shape parameter.
    m_vg : Float
        Van Genuchten shape parameter.    
    W : Dolfin function space
        Function space onto which saturated water content values are projected.

    Returns
    -------
    theta_s : Dolfin function
        Saturated water content.

    """
    
    # Initiating function.
    theta_s = Function(W)
    
    # Setting nodal values.
    theta_s.vector()[:] = vec_theta_s(status, h.vector()[:], h_.vector()[:], theta_.vector()[:], alpha_vg.vector()[:], alpha_vg_d, theta_r0, theta_s0, n_vg, m_vg)
    
    return theta_s

# Defining a function for the hysteretic water content:
def theta_hyst(h, alpha_vg, theta_r, theta_s, n_vg, m_vg):
    """
    Function for hysteretic water contents.

    Parameters
    ----------
    h : Dolfin function.
        Pressure head at current timestep.
    alpha_vg : Dolfin function.
        Inverse of air entry pressure head at current timestep.
    theta_r : Dolfin function.
        Residual water content at current timestep.
    theta_s : Dolfin function.
        Saturated water content at current timestep.
    n_vg : Float.
        Van Genuchten shape parameter.
    m_vg : Float.
        Van Genuchten shape parameter.
        
    Returns
    -------
    theta : Dolfin function
        Water content

    """
    return theta_r + (theta_s - theta_r)*se(h, alpha_vg, n_vg, m_vg)

# Defining a function for the inverse of water content. Used to ensure that
# initial water content is the same whether rhizodeposits are included or not.
def inv_theta_hyst(theta, alpha_vg, theta_r, theta_s, n_vg, m_vg):
    """
    Inverse of water content function.

    Parameters
    ----------
    theta : Float.
        Water content
    alpha_vg : Float.
        Inverse of air entry pressure head at current timestep.
    theta_r : Float.
        Residual water content at current timestep.
    theta_s : Float.
        Saturated water content at current timestep.
    n_vg : Float.
        Van Genuchten shape parameter.
    m_vg : Float.
        Van Genuchten shape parameter.
        
    Returns
    -------
    h : Float
        Pressure head 

    """
    
    inv = -(1/alpha_vg)*pow(pow((theta_s - theta_r)/(theta - theta_r), 1/m_vg) - 1, 1/n_vg)
    
    return inv

# Defining a function for hysteretic water content that is defined using the 
# .vector()[:]
def vec_theta_hyst(h, alpha_vg, theta_r, theta_s, n_vg, m_vg, W):
    """
    Function for hysteretic function defined using .vector()[:]

    Parameters
    ----------
    h : Dolfin function of space 
        Pressure head.
    alpha_vg : Dolfin function of space
        Inverse of air entry pressure head.
    theta_r : Dolfin function of space
        Residual water content.
    theta_s : Dolfin function of space
        Saturated water content.
    n_vg : Float.
        Van Genuchten shape parameter.
    m_vg : Float.
        Van Genuchten shape parameter.
    W : Dolfin function space.
        Space onto which the nodal values of the water content are projected.

    Returns
    -------
    theta : Dolfin function of space 
        hysteretic water content

    """
    theta = Function(W)
    
    theta.vector()[:] = theta_r.vector()[:] + (theta_s.vector()[:] - theta_r.vector()[:])*se(h.vector()[:], alpha_vg.vector()[:], n_vg, m_vg)
    
    return theta

# Defining a function for effective saturation that is defined using .vector()[:]
def vec_se_hyst(h, alpha_vg, n_vg, m_vg, W):
    """
    Function for hysteretic effective saturation defined using .vector()[:]

    Parameters
    ----------
    h : Dolfin function of space
        Pressure head.
    alpha_vg : Dolfin function of space
        Inverse of air entry pressure head.
    n_vg : Dolfin function of space
        Van Genuchten shape parameter.
    m_vg : Dolfin function of space
        Van Genuchten shape parameter.
    W : Dolfin function space.
        Function space onto which effective saturation should be projected    

    Returns
    -------
    output : Doflin function of space
        Effective saturation.

    """
    output = Function(W)
    
    output.vector()[:] = se(h.vector()[:], alpha_vg.vector()[:], n_vg, m_vg)
    
    return output

# Defining a function for the nodal saturated hydraulic conductivity.
def nodal_Ks(status, h, h_, Ks_, alpha_vg_d, Ksd, Ksw, a_st, b_st, a_ca, b_ca, a_vis, b_vis, cw, cd, beta):
    """
    

    Parameters
    ----------
    status : Float
        Indication of whether the soil is wetting or drying.
    h : Float
        Pressure head at current step.
    h_ : Float
        Pressure head at previous step.
    Ks_ : Float
        Saturated hydraulic conductivity at previous step.
    alpha_vg_d : Float
        Base value for inverse of air entry pressure head when drying.
    Ksd : Float
        Base drying saturated hydraulic conductivity.
    Ksw : Float
        Base wetting saturated hydraulic conductivity.
    a_st : Float
        First parameter in surface tension as function of concentration of 
        rhizodeposits in solution.
    b_st :  Float
        Second parameter in surface tension as function of concentration of
        rhizodeposits in solution.
    a_ca : Float
        First parameter in contact angle as function of dried rhizodeposit 
        concentration.
    b_ca : Float
        Second parameter in contact angle as function of dried rhizodeposit 
        concentration.
    a_vis : Float
        First parameter in function for relation between concentration of 
        rhizodeposits in the solution and solution viscosity.
    b_vis : Float
        Second parameter in function for relation between concentration of 
        rhizodeposits in the solution and solution viscosity.
    cw : Float
        Suspended rhizodeposit concentration.
    cd : Float
        Dried rhizodeposit concentration.
    beta : Float
        Power to which the fraction of water surface tension by rhizodeposit
        solution surface tension is raised when multiplying the saturated
        hydraulic conductivity.

    Returns
    -------
    Ks : Float
        Saturated hydraulic conductivity.

    """
    
    # Surface tension of water.
    gamma0 = rhiz_surf_tension(0, a_st, b_st)
    
    # Contact angle on pore surface without rhizodeposit.
    omega_w0 = rhiz_contact_angle(0, a_ca, b_ca)
    
    # Viscosity of soil water solution without rhizodeposits
    mu0 = rhiz_viscosity(0, a_vis, b_vis)
    
    # Surface tension of rhizodeposit solution
    gamma = rhiz_surf_tension(cw, a_st, b_st)
    
    # Contact angle on pore surface with dried rhizodeposit.
    omega_w = rhiz_contact_angle(cd, a_ca, b_ca) 
    
    # Viscosity of rhizodeposit solution.
    mu = rhiz_viscosity(cw, a_vis, b_vis)
    
    # If pressure head at current previous step
    # are non-negative, then we set saturated hydraulic conductivity to the
    # value from the previous timestep.
    if h >= 0.0 or h_ >= 0.0:
        Ks = Ks_
    
    # When we are in a drying curve.
    elif status == alpha_vg_d:
        Ks = Ksd*pow(gamma0/gamma, beta)*(cos(omega_w)/cos(omega_w0))*(mu0/mu)
        
    # When we are in a wetting curve.
    else:
        Ks = Ksw*pow(gamma0/gamma, beta)*(cos(omega_w)/cos(omega_w0))*(mu0/mu)
    
    return Ks

# Vectorizing nodal Ks function
vec_Ks = np.vectorize(nodal_Ks)

# Regularisation constant for hydraulic conductivity function.
eps = 1E-15

# Tortuosity of porous medium.
l = 0.5

# Defining a python function to create a dolfin function of Ks.
def Ks_func(status, h, h_, Ks_, alpha_vg_d, Ksd, Ksw, a_st, b_st, a_ca, b_ca, a_vis, b_vis, cw, cd, beta, W):
    """
    

    Parameters
    ----------
    status : np.array(N_scalar_nodes,)
        Vector displaying the wetting and drying status at each node.
    h : Dolfin function
        Pressure head at current step.
    h_ : Dolfin function
        Pressure head at previous step.
    Ks_ : Dolfin function
        Saturated hydraulic conductivity at previous step.
    alpha_vg_d : Float
        Base value for inverse of air entry pressure head when drying.
    Ksd : Dolfin function
        Base drying saturated hydraulic conductivity.
    Ksw : Dolfin function
        Base wetting saturated hydraulic conductivity.
    a_st : Float
        First parameter in surface tension as function of suspended rhizodeposit 
        concentration.
    b_st :  Float
        Second parameter in surface tension as function of suspended rhizodeposit 
        concentration.
    a_ca : Float
        First parameter in contact angle as function of dried rhizodeposit 
        concentration.
    b_ca : Float
        Second parameter in contact angle as function of dried rhizodeposit 
        concentration.
    a_vis : Float
        First parameter in function for relation between concentration of 
        rhizodeposits in the solution and solution viscosity.
    b_vis : Float
        Second parameter in function for relation between concentration of 
        rhizodeposits in the solution and solution viscosity.
    cw : Dolfin function
        Suspended rhizodeposit concentration.
    cd : Dolfin function
        Dried rhizodeposit concentration.
    beta : Float
        Power to which the fraction of water surface tension by rhizodeposit
        solution surface tension is raised when multiplying the saturated
        hydraulic conductivity.
    W : Dolfin function space
        Function space onto which nodal values are projected.

    Returns
    -------
    Ks : Dolfin function
        Function for saturated hydraulic conductivity.

    """
         
    Ks = Function(W)
                                                              
    Ks.vector()[:] = vec_Ks(status, h.vector()[:], h_.vector()[:], Ks_.vector()[:], alpha_vg_d, Ksd, Ksw, a_st, b_st, a_ca, b_ca, a_vis, b_vis, cw.vector()[:], cd.vector()[:], beta)
    
    return Ks

# Defining regularised relative conductivity function.
def Kr(h, alpha_vg, n_vg, m_vg, theta_r0, theta_s0):
    """
    

    Parameters
    ----------
    h : Float.
        Pressure head.
    alpha_vg : Float 
        The shape parameter for inverse of air entry pressure head to be used 
        in the Van Genuchten function. Based on whether the soil at that node 
        is wetting or drying.
    n_vg : Float
        Van Genuchten shape parameter.
    m_vg : Float
        Van Genuchten shape parameter.
    theta_r0 : Float
        Base value of residual water content.
    theta_s0 : Float
        Base value of saturated water content.

    Returns
    -------
    Kr : Float
        Relative hydraulic conductivity.

    """
    
    return pow((2.0*(theta_s0 - theta_r0)*se(h, alpha_vg, n_vg, m_vg) + eps*(1.0 - 2.0*se(h, alpha_vg, n_vg, m_vg)))/(2.0*(theta_s0 - theta_r0)), l)*pow(1.0 - pow(1.0 - pow((2.0*(theta_s0 - theta_r0)*se(h, alpha_vg, n_vg, m_vg) + eps*(1.0 - 2.0*se(h, alpha_vg, n_vg, m_vg)))/(2.0*(theta_s0 - theta_r0)), 1.0/m_vg), m_vg), 2.0) 

# Defining ficticious parameter to ensure hydraulic conductivity scanning 
# trajectories pass through regularised hydraulic conductivity value at 
# previous timestep.
def nodal_Kstar(h, h_, K_, Ks, alpha_vg, alpha_vg_d, Kr_max, status, n_vg, m_vg, theta_r0, theta_s0):
    """
    

    Parameters
    ----------
    h : Float
        Current  pressure head.
    h_ : Float
        Previous pressure head.
    K_ : Float
        Previous hysdraulic conductivity.
    Ks : FLoat
        Current saturated hydraulic conductivity.
    alpha_vg : Float
        Current inverse air entry pressure head.
    alpha_vg_d : Float
        Base value for inverse of air entry pressure head when drying.
    Kr_max : Float
        Maximum possible value of regularised relative hydraulic conductivity.
    status : Float
        Current wetting/drying status.
    n_vg : Float
        Van Genuchten shape parameter.
    m_vg : Float
        Van Genuchten shape parameter.
    theta_r0 : Float
        Base value of residual water content.
    theta_s0 : Float
        Base value of saturated water content.

    Returns
    -------
    Kstar : Float
        Ficticious parameter.

    """
    
    
    # If we are drying then we do not need the ficticious parameter
    if status == alpha_vg_d or h >= 0.0 or h_ >= 0.0 or Ks*Kr(h_, alpha_vg, n_vg, m_vg, theta_r0, theta_s0) - Ks*Kr_max == 0.0:
        Kstar = 0.0
        
    # If we are wetting then the ficticious parameter is defined as follows.
    else:
        Kstar = Ks*Kr_max*(1 - (K_-Ks*Kr_max)/(Ks*Kr(h_, alpha_vg, n_vg, m_vg, theta_r0, theta_s0) - Ks*Kr_max))

    return Kstar

# Vectorizing nodal Kstar function.
vec_Kstar = np.vectorize(nodal_Kstar)

# Defining a python function to create a dolfin function of Kstar.
def Kstar_func(h, h_, K_, Ks, alpha_vg, alpha_vg_d, Kr_max, status, n_vg, m_vg, theta_r0, theta_s0, W):
    """
    

    Parameters
    ----------
    h : Dolfin function
        Pressure head at current step.
    h_ : Dolfin function
        Pressure head from previous step.
    K_ : Dolfin function
        Hydraulic conductivity from previous step.
    Ks : Dolfin function
        Current saturated hydraulic conductivity.
    alpha_vg : Dolfin function
        Current inverse air entry pressure head.
    alpha_vg_d : Float
        Base value for inverse of air entry pressure head when drying.
    Kr_max : Float
        Maximum possible value of regularised relative hydraulic conductivity.
    status : np.array(N_scalar_nodes, )
        Vector of wetting and drying status at each of the nodes.
    n_vg : Float
        Van Genuchten shape parameter.
    m_vg : Float
        Van Genuchten shape parameter.
    theta_r0 : Float
        Base value of residual water content.
    theta_s0 : Float
        Base value of saturated water content.
    W : Dolfin function space
        Function space onto which the values.

    Returns
    -------
    Kstar : Dolfin function
        Function for ficticious parameter in the hydraulic conductivity.

    """
        
    Kstar = Function(W)
    
    Kstar.vector()[:] = vec_Kstar(h.vector()[:], h_.vector()[:], K_.vector()[:], Ks.vector()[:], alpha_vg.vector()[:], alpha_vg_d, Kr_max, status, n_vg, m_vg, theta_r0, theta_s0)

    return Kstar

# Defining scaling parameter to ensure hydraulic conductivity scanning
# trajectories pass through regularised hydraulic conductivity value at
# previous timestep.
def nodal_alphaK(h, h_, K_, Ks, alpha_vg, alpha_vg_d, Kr_max, status, n_vg, m_vg, theta_r0, theta_s0):
    """
    

    Parameters
    ----------
    h : Float
        Pressure head at current step.
    h_ : Float
        Pressure head from previous step.
    K_ : Float
        Hydraulic conductivity from previous step.
    Ks : FLoat
        Current saturated hydraulic conductivity.
    alpha_vg : Float
        Current inverse air entry pressure head.
    alpha_vg_d : Float
        Base value for inverse of air entry pressure head when drying.
    Kr_max : Float
        Maximum possible value of regularised relative hydraulic conductivity.
    status : Float
        Indicator of whether we are wetting or drying at current step.
    n_vg : Float
        Van Genuchten shape parameter.
    m_vg : Float
        Van Genuchten shape parameter.
    theta_r0 : Float
        Base value of residual water content.
    theta_s0 : Float
        Base value of saturated water content.
        
    Returns
    -------
    alpha_K : Float
        Scaling parameter in hysteretic hydraulic conductivity function.

    """
    
    # If pressure head from previous timestep is greater than or equal to zero
    if h >= 0.0 or h_ >= 0.0 or Ks*Kr(h_, alpha_vg, n_vg, m_vg, theta_r0, theta_s0) - Ks*Kr_max == 0.0 or Ks*Kr(h_, alpha_vg, n_vg, m_vg, theta_r0, theta_s0) == 0.0:
        alphaK = 1.0
    
    elif status == alpha_vg_d:
        alphaK = K_/(Ks*Kr(h_, alpha_vg, n_vg, m_vg, theta_r0, theta_s0))

    else:
        alphaK = (K_- Ks*Kr_max)/(Ks*Kr(h_, alpha_vg, n_vg, m_vg, theta_r0, theta_s0) - Ks*Kr_max)
        
    return alphaK

# Vectorizing nodal function
vec_alphaK = np.vectorize(nodal_alphaK)

# Python function to return a dolfin function for the alphaK parameter in the
# hysteretic hydraulic conductivity function.
def alphaK_func(h, h_, K_, Ks, alpha_vg, alpha_vg_d, Kr_max, status, n_vg, m_vg, theta_r0, theta_s0, W):
    """
    

    Parameters
    ----------
    h : Dolfin function
        Pressure head at current linearisation iteration.
    h_ : Dolfin function
        Pressure head at previous timestep.
    K_ : Dolfin function
        Hydraulic conductivity at previous timestep.
    Ks : Dolfin function
        Current saturated hydraulic conductivity.
    alpha_vg : Dolfin function
        Current inverse air entry pressure head.
    alpha_vg_d : Float
        Base value for inverse of air entry pressure head when drying.
    Kr_max : Float
        Maximum possible value of regularised relative hydraulic conductivity.
    status : np.array(N_scalar_nodes,)
        Wetting and drying status.
    n_vg : Float
        Van Genuchten shape parameter.
    m_vg : Float
        Van Genuchten shape parameter.
    theta_r0 : Float
        Base value of residual water content.
    theta_s0 : Float
        Base value of saturated water content.
    W : Dolfin function space
        Function space onto which the values of alphaK are projected.

    Returns
    -------
    alphaK : Dolfin function
        Function for alphaK parameter.

    """
    
    alphaK = Function(W)
    
    alphaK.vector()[:] = vec_alphaK(h.vector()[:], h_.vector()[:], K_.vector()[:], Ks.vector()[:], alpha_vg.vector()[:], alpha_vg_d, Kr_max, status, n_vg, m_vg, theta_r0, theta_s0)
    
    return alphaK

# Defining a Python function that returns a Dolfin function for the hydraulic
# conductivity.
def K_hyst(h, Ks, Kstar, alphaK, alpha_vg, n_vg, m_vg, theta_r0, theta_s0):
    """
    

    Parameters
    ----------
    h : Dolfin function
        Pressure head.
    Ks : Dolfin function
        Saturated hydraulic conductivity.
    Kstar : Dolfin function
        Hysteresis parameter 1.
    alphaK : Dolfin function
        Hysteresis parameter 2.
    alpha_vg : Dolfin function
        Inverse air entry pressure head.
    n_vg : Float
        Van Genuchten shape parameter.
    m_vg : Float
        Van Genuchten shape parameter.
    theta_r0 : Float
        Base value of residual water content.
    theta_s0 : Float
        Base value of saturated water content.
          
    Returns
    -------
    K : Dolfin function
        The hysteretic hydraulic conductivity function at the current timestep

    """
    
    return Kstar + alphaK*Ks*Kr(h, alpha_vg, n_vg, m_vg, theta_r0, theta_s0)

# Writing a function for the hysteretic hydraulic conductivity that is defined
# using .vector()[:].
def vec_K_hyst(h, Ks, Kstar, alphaK, alpha_vg, n_vg, m_vg, theta_r0, theta_s0, W):
    """
    Function for hysteretic hydraulic conductivity that is written using the 
    .vector()[:] method.

    Parameters
    ----------
    h : Dolfin function of space
        Pressure head.
    Ks : Dolfin function of space
        Saturated hydraulic conductivity.
    Kstar : Dolfin function of space
        Hysteresis parameter 1.
    alphaK : Dolfin function of space
        Hysteresis parameter 2.
    alpha_vg : Dolfin function of space
        Inverse of air entry pressure head.
    n_vg : Float
        Van Genuchten shape parameter.
    m_vg : Float
        Van Genuchten shape parameter.
    theta_r0 : Float
        Base value of residual water content.
    theta_s0 : Float
        Base value of saturated water content.
    W : Dolfin function space
        Space onto which the nodal values of hysteretic hydraulic conductivity
        are to be projected.

    Returns
    -------
    K : Dolfin function 
        The hysteretic hydraulic conductivity function at the current timestep

    """
    K = Function(W)
    
    K.vector()[:] = Kstar.vector()[:] + alphaK.vector()[:]*Ks.vector()[:]*Kr(h.vector()[:], alpha_vg.vector()[:], n_vg, m_vg, theta_r0, theta_s0)
    
    return K

###############################################################################
# Defining a function for returning the flux at each timestep
###############################################################################

def q(K, h, V, e3):
    """
    Vector function for water flux.

    Parameters
    ----------
    K : Dolfin function
        Hydraulic conductivity.
    h : Dolfin function
        Pressure head.
    V : Dolfin vector function space
        Vector function space onto which the water flux will be projected.
    e3 : Canonical vector in positive vertical direction.    

    Returns
    -------
    q : Dolfin vector function
        The water flux at the current step

    """
    
    q = project(-K*(grad(h) + e3), V)
    
    return q

def theta_prime(h, alpha_vg, theta_r, theta_s, n_vg):
    """
    A function which returns the derivative of the Van Genuchten soil
    water content function with respect to pressure head.

    Parameters
    ----------
    h : Float
        Soil water pressure head.
        
    alpha_vg : Float
        Inverse of air entry pressure head (wetting or drying value).
    
    theta_r : Float
        Residual water content.
        
    theta_s : Float
        Saturated water content.
        
    n_vg : Float
        Van Genuchten shape parameter.
    
    Returns
    -------
    theta_prime : Float
        Derivative of theta applied to soil water pressure head h.

    """
    output = pow(alpha_vg, n_vg)*(1 - n_vg)*(theta_s - theta_r)*h*pow(abs(h), n_vg - 2)*pow(1 + pow(abs(alpha_vg*h), n_vg), (1 - 2*n_vg)/n_vg)
    
    return output

# Defining a function which returns a tight upper bound for the Lipschitz
# constant of the water retention function when given minimum and maximum
# pressure head values.
def L_retention_nodal(alpha_vg, theta_r, theta_s, n_vg, h_min, h_max):
    """
    

    Parameters
    ----------
    alpha_vg : Float
        Alpha parameter in Van Genuchten formula (inverse of the air entry 
        pressure head).
    theta_r : Float
        Residual water content.
    theta_s : Float
        Saturated water content.
    n_vg : Float
        Van Genuchten shape parameter.
    h_min : Float
        Lowest value of pressure head at which to check for Lipschitz constant
    h_max : Float
        Highest value of pressure head at which to check for Lipschitz constant
            
    Returns
    -------
    L_theta: Float
        Tight upper bound for the Lipschitz constant of the water retention 
        function.

    """
    
    # Range of soil water pressure head values over which the maximum of the 
    # derivative of the water content is likely to occur.
    # h = np.linspace(h_min, h_max, 1000000)
    h = np.linspace(h_min, h_max, 1000)
    
    # Output is then the ceiling (2dp) of the maximum of the derivative
    # accross this range.
    output = np.ceil(10000.0*np.max(theta_prime(h, alpha_vg, theta_r, theta_s, n_vg)))/10000.0
    
    return output

L_retention = np.vectorize(L_retention_nodal)

# Theta prime function for use in the newton method.
def theta_prime_newt(h, alpha_vg, theta_r, theta_s, n_vg):
    """
    

    Parameters
    ----------
    h : Dolfin function of space.
        Pressure head.
    alpha_vg : Dolfin function of space.
        Inverse of air entry pressure head.
    theta_r : Dolfin function of space.
        Residual water content.
    theta_s : Dolfing function of space. 
        Saturated water content.
    n_vg : Float.
        Van Genuchten shape parameter.

    Returns
    -------
    theta_prime : Dolfin function
        Dericative of water content function with repsect to pressure head.

    """
    
    theta_prime = pow(alpha_vg, n_vg)*(1 - n_vg)*(theta_s - theta_r)*h*pow(abs(h), n_vg - 2)*pow(1 + pow(abs(alpha_vg*h), n_vg), (1 - 2*n_vg)/n_vg)
    
    return theta_prime

# Theta prime function for use in the newton method.
def vec_theta_prime_newt(h, alpha_vg, theta_r, theta_s, n_vg, W):
    """
    

    Parameters
    ----------
    h : Dolfin function of space.
        Pressure head.
    alpha_vg : Dolfin function of space.
        Inverse of air entry pressure head.
    theta_r : Dolfin function of space.
        Residual water content.
    theta_s : Dolfin function of space. 
        Saturated water content.
    n_vg : Float.
        Van Genuchten shape parameter.
    W : Dolfin function space.
        Function space onto which the nodal values of theta_prime are projected

    Returns
    -------
    theta_prime : Dolfin function
        Dericative of water content function with repsect to pressure head.

    """
    
    theta_prime = Function(W)
    
    theta_prime.vector()[:] = pow(alpha_vg.vector()[:], n_vg)*(1 - n_vg)*(theta_s.vector()[:] - theta_r.vector()[:])*h.vector()[:]*pow(abs(h.vector()[:]), n_vg - 2)*pow(1 + pow(abs(alpha_vg.vector()[:]*h.vector()[:]), n_vg), (1 - 2*n_vg)/n_vg)
    
    return theta_prime

###############################################################################
# Setting parameter values for uptake, evaporation and runoff functions
###############################################################################

# Fraction of soil surface covered by vegetation
# (Allen et al 1998 Chapter 7, Table 21). Young wheat plant
fc = 0.1

# Fraction of soil surface wetted by precipitation
# (Allen et al 1998 Chapter 7, Table 20).
fw = 1.0

# Fraction of soil surface that is exposed and wetted
# (Allen et al 1998 Chapter 7).
few = np.amin(np.array([1.0-fc, fw]))

# Reference evapotranspiration for humid subtropical climate,
# cmd^(-1), (Allen et al 1998).
ET0 = 0.1

# Value used for "low" transpiration rate (LT^{-1}). Default value is in metres 
# per day (Cai et al 2018).
T3l = 9.6E-2

# Value used for "high" transpiration rate (LT^{-1}). Default value is in 
# metres per day (Cai et al 2018).
T3h = 0.48

# Critical water stress index, (-), (Cai et al 2018). Determines the extent to 
# which compensated uptake occurs. 
wc = 0.8

###############################################################################
# Defining function for value of basal crop coefficient
###############################################################################

# Function defined using infor in Chapter 3: Winter and spring wheat growth 
# stages by Hall R, Nleya T 2019
def Kcb_ini_func(age, Kcb_in_1, Kcb_in_2):
    """
    Function for the initial basal crop coefficient of wheat given the plant's 
    age.

    Parameters
    ----------
    age : Float > 0, <= 30
        Age of plant in days .
    Kcb_in_1 : Float
        Initial basal crop coefficient 1.
    Kcb_in_2 : Float
        Initial basal crop coefficient 2.

    Returns
    -------
    Kcb : Float
        Basal crop coefficient.

    """
    if age > 0 and age <= 6: 
        Kcb = (Kcb_in_1/6)*age
    
    elif age > 6 and age <= 30:
        Kcb = ((Kcb_in_2 - Kcb_in_1)/24)*(age - 6) + Kcb_in_1
    
    else:
        raise TypeError('The age entered is invalid. Must be >0, <=30')
    
    return Kcb

###############################################################################
# Defining the function for root water uptake. 
###############################################################################

# Defining a function of water content for the water stress response at a given 
# node.
def nodal_alpha_f(theta, theta1, theta2, theta3, theta4):
    """
    A function that returns the water stress response at a given node.
    
    Parameters
    ----------
    theta : Float
        Soil water content.
        
    theta1 : Float > 0
        water content associated with plant anaerobiosis, 
        due to soil saturation (L). 
        
    theta2: Float > 0
        Upper limit of water content threshold 
        within which RWU is assumed maximum. 
        
    theta3: Float > 0
        Lower limit of water content threshold 
        within which RWU is assumed maximum(L).
        
    theta4: Float > 0
        water content associated with plant wilting point (L). 
    
    Returns
    -------
    stress_response : Float in [0, 1]
            Water stress response (-).
            If unstressed then stress_response = 1.
            If stressed then stress_response < 1.
    """
    # print(theta)
    
    if theta >= theta1 or theta < theta4:
        stress_response = 0.0
    
    elif theta >= theta2 and theta < theta1:
        stress_response = (theta - theta1)/(theta2 - theta1)
    
    elif theta >= theta3 and theta < theta2:
        stress_response = 1.0
        
    elif theta >= theta4 and theta < theta3:
        stress_response = (theta - theta4)/(theta3 - theta4)
        
    else:
        print('I cannot generate a stress response for this value of water content', theta)
        
    return stress_response

# Vectorising the pointwise water stress response function
vec_alpha_f = np.vectorize(nodal_alpha_f)

# Defining a python function to return a dolfin function for the alpha_f 
# parameter in the uptake function.                                             
def alpha_f_func(theta, theta1, theta2, theta3, theta4, W):
    """
    

    Parameters
    ----------
    theta : Dolfin function
        Water content at given step
    theta1 : Float > 0
        water content associated with plant anaerobiosis, 
        due to soil saturation (L). 
    theta2: Float > 0
        Upper limit of water content threshold 
        within which RWU is assumed maximum. 
    theta3: Float > 0
        Lower limit of water content threshold 
        within which RWU is assumed maximum(L).
    theta4: Float > 0
        water content associated with plant wilting point (L). 
    W : Dolfin function space
        Function space onto which the alpha_f parameter will be projected.

    Returns
    -------
    alpha_f : Dolfin function
        alpha_f parameter in uptake function.

    """
    
    alpha_f = Function(W)
    
    alpha_f.vector()[:] = vec_alpha_f(theta.vector()[:], theta1, theta2, theta3, theta4)
    
    return alpha_f
    
# Defining a function that computes the compensation factor in the uptake 
# function
def compensation(theta, theta1, theta2, theta3, theta4, NRLD, W, dx):
    """
    

    Parameters
    ----------
    theta : Dolfin function
        Water content at given step
    theta1 : Float > 0
        water content associated with plant anaerobiosis, 
        due to soil saturation (L). 
    theta2: Float > 0
        Upper limit of water content threshold 
        within which RWU is assumed maximum. 
    theta3: Float > 0
        Lower limit of water content threshold 
        within which RWU is assumed maximum(L).
    theta4: Float > 0
        water content associated with plant wilting point (L). 
    NRLD : Dolfin function
        Normalised root length density.
    W : Dolfin function space
        Function space onto which stress vector is projected before integration
    dx: Dolfin measure.
        The measure associated with the mesh upon which the scalar function
        space W is defined.
    
    Returns
    -------
    compensation : Float
        Compensation factor in uptake function.

    """
    stress = Function(W)
    stress_index_integrand = Function(W)
    
    # Computing water stress response.
    stress.vector()[:] = vec_alpha_f(theta.vector()[:], theta1, theta2, theta3, theta4)    
    
    # Computing the integrand involved in computing the water stress index
    stress_index_integrand.vector()[:] = stress.vector()[:]*NRLD.vector()[:]
    
    # Computing the compensation factor.
    compensation = 1/max(assemble(stress_index_integrand*dx), wc)
    
    return compensation

# Defining a python function that returns a dolfin function for compensated
# root water uptake.
def S(Tp, compensation, alpha_f, NRLD):
    """
    

    Parameters
    ----------
    Tp : Float
        Transpiration rate of plant.
    compensation : Float
        Compensation of uptake due to water stress.
    alpha_f : Dolfin function
        Water stress being experienced by the plant.
    NRLD : Dolfin function
        Normalised root length density.

    Returns
    -------
    S : Dolfin function
        Compensated root water uptake

    """

    S = Tp*compensation*alpha_f*NRLD
    
    return S

# Defining a function of water content for 1D compensated water uptake rate
# that is defined using .vector(){:}
def vec_S(W, theta, theta1, theta2, theta3, theta4, Tp, NRLD, dx):
    """
    Function for compensated root water uptake.
    Parameters
    ----------
    W: Dolfin  scalar function space.
    
    theta : Dolfin function. 
        Soil water content.
        
    theta1 : Float > 0
        water content associated with plant anaerobiosis, 
        due to soil saturation (L). 
        
    theta2: Float > 0
        Upper limit of water content threshold 
        within which RWU is assumed maximum. 
        
    theta3: Float > 0
        Lower limit of water content threshold 
        within which RWU is assumed maximum(L).
        
    theta4: Float > 0
        water content associated with plant wilting point (L).
        
    Tp: Float > 0
        Plant evapotranspiration.
    
    NRLD: Dolfin function. 
        Normalised root length density (L^(-3)). 
        
    dx: Dolfin measure.
        The measure associated with the mesh upon which the scalar function
        space W is defined.
        
    Returns
    -------
    S : Dolfin function
        Root water uptake (T^{-1})         
    """
    
    # Initiating functions for root water uptake and water stress response.
    S = Function(W)
    stress = Function(W)
    stress_index_integrand = Function(W)
    
    # Computing water stress response.
    stress.vector()[:] = vec_alpha_f(theta.vector()[:], theta1, theta2, theta3, theta4)    
    
    # Computing the integrand involved in computing the water stress index
    stress_index_integrand.vector()[:] = stress.vector()[:]*NRLD.vector()[:]
    
    # Computing the compensation factor.
    compensation = 1/max(assemble(stress_index_integrand*dx), wc)
    
    # Computing the root water uptake rate.
    S.vector()[:] = Tp*compensation*stress.vector()[:]*NRLD.vector()[:]
    
    return S

###############################################################################
# Defining the function for evaporation from the soil surface
###############################################################################

# Defining a nodal function for the evaporation coefficent
# (Allen et al 1998 Chapter 7).
def nodal_Ke(theta, Kcmax, Kcb, theta_fc, theta_wp):
    """
    A function for the evaporation from a given node on the soil surface.
    
    Parameters
    ----------
    theta : Float 
        Soil water content.
    Kcmax : FLoat
        Maximum crop coefficient.
    Kcb : Float
        Basal crop coefficient.    
    theta_wp : Float
        Wilting point.
    theta_fc : Float
        Field capacity.
    
    Returns
    -------
    Ke : Float 
        evaporation coefficient (-).         
    """
    
    # Computing the reduction coefficient.
    if theta > theta_fc:
        Kr = 1
    elif theta < 0.5*theta_wp:
        Kr = 0
    else:
        Kr = (theta - 0.5*theta_wp)/(theta_fc - 0.5*theta_wp) 
    
    # Computing the evaporation coefficient.
    Ke_h = np.amin(np.array([Kr*(Kcmax - Kcb), few*Kcmax]))
    
    return Ke_h

# Vectorising the function for the evaporation coefficient.
vec_Ke = np.vectorize(nodal_Ke)

# Defining a python function to return a dolfin function for Ke.
def Ke_func(theta, Kcmax, Kcb, theta_fc, theta_wp, W):
    """
    

    Parameters
    ----------
    theta : Dolfin function
        Water content.
    Kcmax : FLoat
        Maximum crop coefficient
    Kcb : Float
        Basal crop coefficient    
    theta_wp : Float
        Wilting point
    theta_fc : Float
        Field capacity
    W : Dolfin function space
        Function space onto which Ke is projected.

    Returns
    -------
    Ke : Dolfin function
        Evaporation coefficient.

    """
    
    Ke = Function(W)
    
    Ke.vector()[:] = vec_Ke(theta.vector()[:], Kcmax, Kcb, theta_fc, theta_wp)
    
    return Ke

# Defining a python function which returns a dolfin function for evaporation
def evaporation(Ke):
    """
    

    Parameters
    ----------
    Ke : Dolfin function
        Evaporation coefficient.

    Returns
    -------
    evaporation : Dolfin function
        Evaporation function.

    """
    return ET0*Ke

# Defining evaporation function, which allows a reduced rate of evaporation
# if it is necessary:
def vec_evaporation(theta, W, Kcmax, Kcb, theta_fc, theta_wp):
    
    """
    Parameters
    ----------
    
    theta : Dolfin function of space and time. 
        Hysteretic soil water content (cm^3cm^(-3)).
    W : Dolfin scalar function space.
    ET0: Float > 0
        Reference evapotranspiration (LT^{-1}).
    Kcmax : FLoat
        Maximum crop coefficient
    Kcb : Float
        Basal crop coefficient    
    theta_wp : Float
        Wilting point
    theta_fc : Float
        Field capacity
        
    Returns
    -------
    evaporation rate: Dolfin function of space and time.
        The evaporation rate (cmd^{-1}).
    """
    # Defining evaporation dolfin form.
    evaporation = Function(W)
    
    # Assigning the nodal values of the evaporation function.
    evaporation.vector()[:] = ET0*vec_Ke(theta.vector()[:], Kcmax, Kcb, theta_fc, theta_wp)
    
    return evaporation

# Defining a function for evaporation with a float input.
def evap_float(surface_theta, Kcmax, Kcb, theta_fc, theta_wp):
    """
    

    Parameters
    ----------
    surface_theta : Float
        Hysteretic soil water content (cm^3cm^(-3)).
    Kcmax : FLoat
        Maximum crop coefficient
    Kcb : Float
        Basal crop coefficient    
    theta_wp : Float
        Wilting point
    theta_fc : Float
        Field capacity

    Returns
    -------
    evap : Float >1
        The evaporation rate (cmd^{-1}).

    """
    
    evap = ET0*nodal_Ke(surface_theta, Kcmax, Kcb, theta_fc, theta_wp)
    
    return evap

###############################################################################
# Defining a function for surface runoff 
###############################################################################

def runoff(theta, theta_s0, precipitation, a):
    """
    A function that computes the runoff from the soil surface.
    
    Parameters
    ----------
    theta : Dolfin function of space and time. 
        Hysteretic soil water content.
        
    theta_s0 : Float.
        Base saturated hydraulic conductivity.
        
    precipitation: Expression object.
        Expression defining the rate of precipitation onto the upper boundary
        during the simulation (LT^{-1}).
        
    a: Float.
        How steep the transition is to complete runoff once the boundary 
        approaches saturation.
    
    Returns
    -------
    runoff : dolfin function of space and time
              The amount of runoff given the current time and pressure head
              value at the soil surface (cmd^{-1})
    """
    
    runoff = precipitation*(1 + (exp(a*(theta - theta_s0)) - 1)/(exp(a*(theta - theta_s0)) + 1))
    
    return runoff

###############################################################################
# Defining function to simultaneously model rhizodeposit transport, 
# hysteretic water transport and root water uptake.
###############################################################################

def simulator(name,  
              soil_type, 
              status0, 
              theta_init, 
              rhizodeposits, 
              Nx, 
              Nt, 
              p_tot, 
              p_pat, 
              p_l1, 
              p_lred, 
              T, 
              tol, 
              cw_init, 
              cd_init,
              ex_total,
              gamma = 2.0,
              imported_mesh = 'def'):
    """
    

    Parameters
    ----------
    name : String
        Specific plant.
    soil_type : String
        Type of soil.
    status0 : String:
        'wetting' or 'drying'
    theta_init : Float:
        Constant water content within domain
    rhizodeposits : Boolean
        Whether or not rhizodeposits and their effect are to be incorporated.
    Nx : Integer
        Level of spatial refinement.
    Nt : Integer
        Number of timesteps.
    p_tot : FLoat
        Total rainfall amount.
    p_pat : Integer
        Number of rainfall events: 1, 2, 3
    p_l1 : Float
        Duration of rainfall event when p_pat == 1.
    p_lred : Float
        Rate at which duration of rainfall event is reduced when increasing the
        total number of rainfall events.
    T : Float
        Final time.
    tol :  Float expressed as power (e.g. 1E-1)
        Difference in consecutive pressure heads that is required to constitute 
        a reversal in wetting/drying direction.
    cw_init : Float or string
        Initial concentration of suspended rhizodeposit. Either a float input
        with the exact desired amount or the string input 'eq' to signify that
        the initial rhizodeposit concentration is at equilibrium.
    cd_init : Float or string
        Initial concentration of dried rhizodeposit. Either a float input
        with the exact desired amount or the string input 'eq' to signify that
        the initial rhizodeposit concentration is at equilibrium.
    ex_total : Float
        Result of summing product of initial water content and initial 
        suspended rhizodeposit concentration with product of soil bulk density and
        dried rhizodeposit concentration
    gamma : Float, optional
        Width of Gaussian support in 3d root length and surface area density
        functions. The default is 2.0.
    imported_mesh: String
        3D mesh that is imported for the construction of root density functions.
        'def' imports the mesh that was constructed using the deprecated
        functionality mshr, 'box' imports the mesh that can still be
        created using available Legacy FEniCS docker images.    

    Returns
    -------
    total_uptake: np.array(Nt, )
        Total root water uptake over simulation time.
    total_evaporation: np.array(Nt, )
        Total evaporation from upper soil surface over simulation time.
    total_deep_percolation: np.array(Nt, )
        Total flux across boundary between rooted zone and fallow zone beneath.
    total_free_drainage: np.array(Nt, )
        Total free drainage through lower soil surface over simulation time.
    total_precipitation: np.array(Nt, )
        Total precipitation landing on soil over simulation.
    total_runoff: np.array(Nt, )
        Total precipitation lost via surface runoff over simulation.
    root_zone_water_content: np.array(Nt, )
        Water content of rooted zone over time.
    Uptake_zone_water_content: no.array(Nt, )
        Water content of uptake zone over time.
    total_upper_flux: np.array(Nt, )
        Total flux accross upper boundary of uptake zone.
    total_atmo_flux: np.array(Nt, )
        Total flux accross upper boundary of soil domain.
    total_lower_flux: np.array(Nt, )
        Total flux accross lower boundary of uptake zone.    

    """
    
    ###########################################################################
    # Creating vectors to store water lifetime data for post processing.
    ###########################################################################
    
    total_precipitation = np.zeros(Nt)
    total_evaporation = np.zeros(Nt)
    total_deep_percolation = np.zeros(Nt)
    total_runoff = np.zeros(Nt)
    total_free_drainage = np.zeros(Nt)
    total_uptake = np.zeros(Nt)
    total_upper_flux = np.zeros(Nt)
    total_atmo_flux = np.zeros(Nt)
    total_lower_flux = np.zeros(Nt)
    root_zone_water_content = np.zeros(Nt)
    uptake_zone_water_content = np.zeros(Nt)
    
    # Creating labels for level of spatial and temporal discretisation. 
    Nx_lab = str(math.modf(Nx)[1])[:-2]
    Nt_lab = str(math.modf(Nt)[1])[:-2]
    
    # Ensuring mesh name entered is valid.
    if not (imported_mesh == 'def' or imported_mesh == 'box'):
        raise TypeError('Invalid entry for imported mesh')
        
    # Ensuring root system name entered is valid.
    if not (name == 'trigo6days' or name == 'trigo15days' or name == 'trigo30days'):
        raise TypeError('Invalid entry for root system name')
    
    # Ensuring soil type entered is valid.
    if not (soil_type == 'sandy_loam' or soil_type == 'loamy_sand' or soil_type == 'sand'):
        raise TypeError('Invalid soil type entered')
        
    # Ensuring initial status entered is valid.
    if not (status0 == 'wetting' or status0 == 'drying'):
        raise TypeError('Invalid initial wetting/drying status entered')
        
    # Ensuring entry for rhizodeposits input is valid.
    if not (rhizodeposits == True or rhizodeposits == False):
        raise TypeError('Invalid entry for rhizodeposits input, should be "True" or "False".')
        
    # Ensuring precipitation pattern is valid.
    if not (p_pat == 3 or p_pat == 2 or p_pat == 1):
        raise TypeError('Invalid precipitation pattern entered.')
        
    # Ensuring that input for initial rhizodeposit concentration is valid.
    if not ((type(cw_init) == float and type(cd_init) == float) or (cw_init == 'eq' and cd_init == 'eq')):
        raise TypeError('Invalid entry for initial dried and saturated rhizodeposits')
        
    # Writing labels to indicate the precipitation pattern being studied on 
    # each of the saved out files.
    
    if p_pat == 3:
        p_tag = '3'
        
    elif p_pat == 2:
        p_tag = '2'
        
    else: 
        p_tag = '1'
        
    # Writing labels to indicate the soil type being considered
    if soil_type == 'sandy_loam':
        st_tag = 'sal'
        
    elif soil_type == 'loamy_sand':
        st_tag = 'los'
        
    elif soil_type == 'sand':
        st_tag = 'san'
        
    if tol == 1E-1:
        tol_tag = 1
    elif tol == 1E-2:
        tol_tag = 2
    elif tol == 1E-3:
        tol_tag = 3
    elif tol == 1E-4:
        tol_tag = 4
    elif tol == 1E-5:
        tol_tag = 5
    elif tol == 1E-6:
        tol_tag = 6
    elif tol == 1E-7:
        tol_tag = 7
    elif tol == 1E-8:
        tol_tag = 8
    elif tol == 1E-9:
        tol_tag = 9
    else:
        tol_tag = 10
                
    # Split the gamma_l values into integer and decimal components for saving
    gamma_l_int = str(math.modf(gamma)[1])[:-2]
    gamma_l_dec = str(np.round(math.modf(gamma)[0], 2))[2:]
    
    # Split the gamma_sa values into integer and decimal components for saving
    gamma_sa_int = str(math.modf(gamma)[1])[:-2]
    gamma_sa_dec = str(np.round(math.modf(gamma)[0], 2))[2:]

    # Writing a label for the final time of the simulation
    T_int = str(math.modf(T)[1])[:-2]
    T_dec = str(np.round(math.modf(T)[0], 2))[2:]

    # Writing a label for the total precipitation
    p_tot_int = str(math.modf(p_tot)[1])[:-2]
    p_tot_dec = str(np.round(math.modf(p_tot)[0], 2))[2:]

    # Writing a label for the precipitation event length for single event
    p_l1_int = str(math.modf(p_l1)[1])[:-2]
    p_l1_dec = str(np.round(math.modf(p_l1)[0], 2))[2:]

    # Writing a label for the reduction rate of precipitation event length 
    # as more events are added
    p_lred_int = str(math.modf(p_lred)[1])[:-2]
    p_lred_dec = str(np.round(math.modf(p_lred)[0], 2))[2:]
    
    # Writing a label for the initial constant pressure head.
    theta_init_int = str(math.modf(theta_init)[1])[:-2]
    theta_init_dec = str(np.round(math.modf(theta_init)[0], 2))[2:]
    
    # Writing a label for the initial suspended rhizodeposit.
    # and for the initial attached rhizodeposit.
    if cw_init == 'eq' and cd_init == 'eq':
        cw_init_int = 'eq'
        cw_init_dec = ''    
        cd_init_int = 'eq'
        cd_init_dec = ''
    
    else:
        cw_init_int = str(math.modf(cw_init)[1])[:-2]
        cw_init_dec = str(np.round(math.modf(cw_init)[0], 2))[2:]
        cd_init_int = str(math.modf(cd_init)[1])[:-2]
        cd_init_dec = str(np.round(math.modf(cd_init)[0], 2))[2:]
    
    mesh_tag = 'glb'
    
    # Writing labels to indicate if root rhizodeposits have been included
    if rhizodeposits == True:
        ex_lab = 'ex'
        
    else:
        ex_lab = 'no_ex'
        
    # Writing labels to indicate if initial state of soil was wetting or drying
    if status0 == 'wetting':
        stat_lab = 'w0'
        
    else:
        stat_lab = 'd0'
    
    ###########################################################################
    # Setting age of plant
    ###########################################################################
    
    if name == 'trigo6days':
        age = 6
    
    elif name == 'trigo15days':
        age = 15
        
    else:
        age = 30
        
    
    ###########################################################################
    # Reading values with which to parameterise expressions for surface tension
    # as a function of solubilised wheat rhizodeposits, contact angle as a 
    # function of attached wheat rhizodeposits and viscosity as a function of
    # solubilised wheat rhizodeposits.
    ###########################################################################
    
    a_st = np.loadtxt('data/st_parameter_values_read_2003.txt')[0]
    b_st = np.loadtxt('data/st_parameter_values_read_2003.txt')[1]
    a_ca = np.loadtxt('data/ca_vs_rhizodeposits_parameter_values_zickenrott2016.txt')[0]
    b_ca = np.loadtxt('data/ca_vs_rhizodeposits_parameter_values_zickenrott2016.txt')[1]
    a_vis = np.loadtxt('data/mu_parameter_values_hosseini_mossaddeghi_2024.txt')[0]
    b_vis = np.loadtxt('data/mu_parameter_values_hosseini_mossaddeghi_2024.txt')[1]
    
    print('surface tension of rhizodeposit solution', rhiz_surf_tension(1.0, a_st, b_st))
    print('surface tension of water', rhiz_surf_tension(0.0, a_st, b_st))
    
    print('contact angle on pore surface with dry rhizodeposits attached', rhiz_contact_angle(6.33E-4, a_ca, b_ca))
    print('contact angle on pore surface without rhizodeposits attached', rhiz_contact_angle(0.0, a_ca, b_ca))
    
    print('Viscosity of rhizodeposit solution', rhiz_viscosity(1.0, a_vis, b_vis))
    print('Viscosity of water', rhiz_surf_tension(0.0, a_vis, b_vis))
    
    # Maximum and minimum values for surface tension.
    gamma_max = rhiz_surf_tension(0.0, a_st, b_st)
    gamma_min = 47.5
    
    # Maximum and minimum values for contact angle.
    omega_max = (35.29/180.0)*pi
    omega_min = rhiz_contact_angle(0.0, a_ca, b_ca)
    
    ###########################################################################
    # Setting soil hydraulic parameters. 
    ###########################################################################
    
    if soil_type == 'sandy_loam':
        
        # Bulk density of sandy loam soil mgcm^{-3}
        # https://hort.ifas.ufl.edu/woody/critical-value.shtml
        rho = 1650
        
        # Residual water content.
        # (Carsel & Parish, 1988)
        theta_r0 = 0.065
        
        # Saturated water content.
        # (Carsel & Parish, 1988)
        theta_s0 = 0.41
        
        # Wetting inverse air entry pressure head cm^{-1} 
        # (Kool and Parker, 1987)
        alpha_vg_w = 0.0521
        
        # Drying inverse air entry pressure head cm^{-1}  
        # (Kool and Parker, 1987)
        alpha_vg_d = 0.0114
        
        # Sandy loam (pore size parameter) 
        # (Carsel & Parish, 1988)
        n_vg = 1.89
        m_vg = 1-1/n_vg
        
        # Drying saturated hydraulic conductivity (cm/day) 
        # (Vogel & Zhang, 1996)
        Ksd = 5.0*24.0
        
        # Wetting saturated hydraulic conductivity (cm/day).
        # (Vogel & Zhang, 1996).
        Ksw = 3.0*24.0
        
        # Wilting point water content, cm^3cm^(-3) or m^3m^(-3), 
        # (Allen 1998, chapter 7 table 19).
        theta_wp = 0.16
        
        # Field capacity water content cm^3cm^(-3) or m^3m^(-3), 
        # (Allen 1998, chapter 7 table 19)
        theta_fc = 0.28
        
    elif soil_type == 'loamy_sand':
        
        # Bulk density of loamy sand mgcm^{-3}
        # https://hort.ifas.ufl.edu/woody/critical-value.shtml
        rho = 1750
        
        # Residual water content.
        # (Carsel & Parish, 1988)
        theta_r0 = 0.057
        
        # Saturated water content.
        # (Carsel & Parish, 1988)
        theta_s0 = 0.41
        
        # Wetting inverse air entry pressure head cm^{-1} 
        # (Carsel & Parrish, 1987)
        alpha_vg_w = 0.248
        
        # Drying inverse air entry pressure head cm^{-1}  
        # (Kool and Parker, 1987)
        alpha_vg_d = 0.124
        
        # Sandy loam (pore size parameter) 
        # (Carsel & Parish, 1988)
        n_vg = 2.28
        m_vg = 1-1/n_vg
        
        # Drying saturated hydraulic conductivity (cm/day) 
        # (Vogel & Zhang, 1996)
        Ksd = 14.59*24.0
        
        # Wetting saturated hydraulic conductivity (cm/day).
        # (Vogel & Zhang, 1996).
        Ksw = 8.754*24.0
        
        # Wilting point water content, cm^3cm^(-3) or m^3m^(-3), 
        # (Allen 1998, chapter 7 table 19).
        theta_wp = 0.1
        
        # Field capacity water content cm^3cm^(-3) or m^3m^(-3), 
        # (Allen 1998, chapter 7 table 19)
        theta_fc = 0.19
        
    elif soil_type == 'sand':
        
        # Bulk density of loamy sand mgcm^{-3}
        # https://hort.ifas.ufl.edu/woody/critical-value.shtml
        rho = 1750
        
        # Residual water content.
        # (Carsel & Parish, 1988)
        theta_r0 = 0.045
        
        # Saturated water content.
        # (Carsel & Parish, 1988)
        theta_s0 = 0.43
        
        # Wetting inverse air entry pressure head cm^{-1} 
        # (Carsel & Parrish, 1987)
        alpha_vg_w = 0.290
        
        # Drying inverse air entry pressure head cm^{-1}  
        # (Kool and Parker, 1987)
        alpha_vg_d = 0.145
        
        # Sandy loam (pore size parameter) 
        # (Carsel & Parish, 1988)
        n_vg = 2.28
        m_vg = 1-1/n_vg
        
        # Drying saturated hydraulic conductivity (cm/day) 
        # (Vogel & Zhang, 1996)
        Ksd = 29.70*24.0
        
        # Wetting saturated hydraulic conductivity (cm/day).
        # (Vogel & Zhang, 1996).
        Ksw = 17.82*24.0
        
        # Wilting point water content, cm^3cm^(-3) or m^3m^(-3), 
        # (Allen 1998, chapter 7 table 19).
        theta_wp = 0.07
        
        # Field capacity water content cm^3cm^(-3) or m^3m^(-3), 
        # (Allen 1998, chapter 7 table 19)
        theta_fc = 0.17
        
    # Defining maximum possible value of regularised relative conductivity.
    Kr_max = pow(1.0 - (eps/2.0)/(theta_s0 - theta_r0), l)*pow(1.0 - pow(1.0 - pow(1.0-(eps/2.0)/(theta_s0 - theta_r0), 1.0/m_vg), m_vg), 2)
    print('Maximum possible value of relative hydraulic conductivity', Kr_max)
    
    ###########################################################################
    # Reading in calibrated parameters for effects of rhizodeposits on soil 
    # hydraulic parameters.
    ###########################################################################
    
    beta = np.loadtxt(f'data/calibrated_parameter_values/beta.txt')
    # print('calibrated beta value = ', beta)
    
    kappa_d = np.loadtxt(f'data/calibrated_parameter_values/kappa_d.txt')
    # print('calibrated gamma drying rate', kappa_d)
    
    kappa_w = np.loadtxt(f'data/calibrated_parameter_values/kappa_w.txt')
    # print('calibrated gamma wetting rate', kappa_w)
    
    ###########################################################################
    # Setting crop parameters
    ###########################################################################
    
    # Basal crop coefficient for early-season Winter Wheat, (-), 
    # (Allen et al 1998 Chapter 7, table 17).
    Kcb_ini_1 = 0.15
    Kcb_ini_2 = 0.5
    Kcb = Kcb_ini_func(age, Kcb_ini_1, Kcb_ini_2)
    # print('Basal crop coefficient =', Kcb)
    
    # Maximum crop coefficient for early-season Winter Wheat, (-)
    # (Allen et al 1998 Chapter 7, page 143).
    Kcmax = 1.30
    
    # Pressure head associated with plant anaerobiosis, due to soil 
    # saturation (cm). (Wesseling 1991, table 5).
    h1 = -0.0
    
    # Setting water content associated with plant anaerobiosis
    theta1 = theta_r0 + (theta_s0 - theta_r0)*se(h1, alpha_vg_w, n_vg, m_vg)
    # print('Water content associated with plant anaerobiosis', theta1)
    
    # Upper limit of pressure head threshold  within which RWU is assumed 
    # maximum (cm). (Wesseling 1991, table 5).
    h2 = -1.0
    
    # Setting water content associated with upper limit of pressure head interval
    # within which RWU is assumed to be maximised.
    theta2 = theta_r0 + (theta_s0 - theta_r0)*se(h2, alpha_vg_w, n_vg, m_vg)
    
    # Value, when transpiration rates are low, of lower limit of pressure 
    # head threshold within which RWU is assumed maximum (cm).
    # Wheat (Wesseling 1991, table 5).
    h3l = -900.0  
    
    # Value, when transpiration rates are high, of lower limit of pressure 
    # head threshold within which RWU is assumed maximum (cm).
    # Wheat (Wesseling 1991, table 5).
    h3h = -500.0  
    
    # Pressure head associated with plant wilting point (cm). Default value is for 
    # Wheat and in metres (Wesseling 1991, table 5).
    h4 = -16000.0
    
    # Water content associated with plant wilting point (L).
    theta4 =  theta_r0 + (theta_s0 - theta_r0)*se(h4, alpha_vg_d, n_vg, m_vg)
    # print('Water content associated with plant wilting point', theta4)
    
    # Rate of rhizodepisit exudation. The suggested interval is taken from
    # (Ptashyk et al., 2011) and expressed as 100-500 pmolg^{-1}rootFWs^{-1}.
    # which converts to 
    # 0.5-2.5 pmolcm^{-2}s^{-1}
    # = 0.5-2.5 * 10^{-12}*86400 molcm^{-2}d^{-1}
    # = 0.5-2.5 * 864 * 10^-10 molcm^{-2}d^{-1}
    # = 0.5-2.5 * 304.3 * 864 * 10^-10 gcm^{-2}d^{-1} (using molar mass of deoxymugineic acid https://pubchem.ncbi.nlm.nih.gov/compound/2_-Deoxymugineic-acid)
    # = 0.5-2.5 * 304.3 * 864* 10^-7 mgcm^-2d^{-1} 
    # Assuming median rate
    r_cw = 1.5*304.3*864*10E-7
    # print('Rate of rhizodeposit production at root surface', r_cw)
        
        
    # Diffusion coefficent of rhizodeposits through pure water
    D_cw0 = 0.65
    
    # Impedance factor due to tortuosity of the pore space
    # Schnepf et al. 2012 Modelling phosphorous uptake by a growing and 
    # exuding root system
    # Zech and de Winter 2023 A Probabilistic Formulation of the Diffusion 
    # Coefficient in Porous Media as Function of Porosity
    tau_imp = 0.3
    
    # Effective diffusion coe
    D_cw = D_cw0*tau_imp
    
    # Transpiration rate, cmd^(-1). 
    Tp = Kcb*ET0
    
    # Determining Lower limit of pressure head threshold within which RWU is 
    # assumed maximum.
   
    # Assuming that Transpiration is high
    h3 = h3h
        
    # Setting water content associated to lower limit of pressure head 
    # threshold within which RWU is assumed maximum.
    theta3 = theta_r0 + (theta_s0 - theta_r0)*se(h3, alpha_vg_d, n_vg, m_vg)
    # print('Lower bound of water content interval in which uptake is not stressed', theta3)
    # print('Upper bound of water content interval in which uptake is not stressed', theta2)
    
    ###########################################################################
    # Importing data for rooting depth and start and end points of uptake zone
    ###########################################################################
    
    # Loading array of segment data of root system.
    segments = np.loadtxt(f'data/{name}formatted.txt')
    
    # Loading rooting depth of root system.
    rooting_depth = float(np.loadtxt(f'data/{name}_rooting_depth.txt'))
    print('rooting depth =', rooting_depth)
    
    # Loading array of start points of uptake zones of each root system.
    uptake_zone_start_points = np.loadtxt(f'data/start_points_start_frac0_1_Nx{Nx}_NRLD1D_gaml{gamma_l_int}_{gamma_l_dec}_mesh_{imported_mesh}.txt')
    uptake_zone_end_points = np.loadtxt(f'data/end_points_start_frac0_1_Nx{Nx}_NRLD1D_gaml{gamma_l_int}_{gamma_l_dec}_mesh_{imported_mesh}.txt')
    
    # Setting uptake zone start point of specific root system being considered
    if name == 'trigo6days':
        uptake_zone_start = uptake_zone_start_points[0]
        uptake_zone_end = uptake_zone_end_points[0]
    
    elif name == 'trigo15days':
        uptake_zone_start = uptake_zone_start_points[1]
        uptake_zone_end = uptake_zone_end_points[1]
        
    else:
        uptake_zone_start = uptake_zone_start_points[2]
        uptake_zone_end = uptake_zone_end_points[2]
    
    ###########################################################################
    # Defining the soil domain that contains the root system
    # and constructing the mesh
    ###########################################################################
    
    # Finding the lowest x_3 value of a segment's end in the architecture.
    minx2 = np.amin(segments[:, 2]) 
    minx6 = np.amin(segments[:, 6])
    minx = np.amin(np.array([minx2, minx6]))
    
    # Finding the highest x_3 value of a segments's end in the architecture.
    maxx2 = np.amax(segments[:, 2])
    maxx6 = np.amax(segments[:, 6])
    maxx = np.amax(np.array([maxx2, maxx6]))
    
    # Saving original x_3 limits.
    minx_old = minx
    maxx_old = maxx
    
    # If the highest x_3 value is above zero then the whole architecture is
    # shifted down by this value.
    if maxx > 0:
        segments[:, 2] -= maxx
        segments[:, 6] -= maxx
        
    # Finding the new lowest x_3 value of a segment's end in the architecture.
    minx2 = np.amin(segments[:, 2]) 
    minx6 = np.amin(segments[:, 6])
    minx = np.amin(np.array([minx2, minx6])) 

    # Finding the new highest x_3 value of a segments's end in the architecture.
    maxx2 = np.amax(segments[:, 2])
    maxx6 = np.amax(segments[:, 6])
    maxx = np.amax(np.array([maxx2, maxx6]))
        
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
    
    # Function space for vector functions.
    V = VectorFunctionSpace(mesh, 'CG', 1)
    
    # Array of nodes for scalar functions
    scalar_nodes = W.tabulate_dof_coordinates()
    
    # print('Vertices', mesh.coordinates()) 
    # print('Nodes', scalar_nodes)

    # Number of nodes for scalar functions.
    N_scalar_nodes = len(scalar_nodes)
    N_vertices = len(mesh.coordinates())
    
    # ###########################################################################
    # # Altering rooting depth so that deep percolation measurement is taken 
    # # slightly above
    # ###########################################################################
    
    # # Rooting depth for deep percolation measurement.
    # rooting_depth_dp = rooting_depth + 0.1*(tp - bttm)
    # print('Point above rooting depth at which deep percolation measurement is taken =', rooting_depth_dp)
    
    ###########################################################################
    # Importing normalised root length density profile and creating root length
    # density profile.
    ###########################################################################
    
    NRLD_nodal = np.loadtxt(f'data/{name}_{Nx}_NRLD1D_gaml{gamma_l_int}_{gamma_l_dec}_node_vals_{imported_mesh}.txt')
    RSA_nodal = np.loadtxt(f'data/{name}_{Nx}_RSA1D_gamsa{gamma_sa_int}_{gamma_sa_dec}_node_vals_{imported_mesh}.txt')
    NRLD = Function(W)
    RSA = Function(W)
    NRLD.vector()[:] = NRLD_nodal
    RSA.vector()[:] = RSA_nodal
    
    # xdmffile_NRLD = XDMFFile(f'data/{name}_plot_1Dnlen_gaml{gamma_l_int}_{gamma_l_dec}_{imported_mesh}.xdmf')
    # xdmffile_RSA = XDMFFile(f'data/{name}_plot_1Drsa_gamsa{gamma_sa_int}_{gamma_sa_dec}_{imported_mesh}.xdmf')
    # xdmffile_NRLD.write(NRLD)
    # xdmffile_RSA.write(RSA)
    
    ###########################################################################
    # Defining the precipitation condition on the upper boundary.
    ###########################################################################
    
    # Defining constant for time variable in the Neumann condition on the
    # upper boundary.
    tt = Constant(0.0)
    
    if p_pat == 1:
        
        # Numerator of precipitation function (rate of precipitation)
        p = p_tot/p_l1
        
        # Centre of bump of precipitation function (halfway point of 
        # precipitation event/delivery of irrigation)
        p_a1 = p_l1/2
        
        # Radius of bump of precipitation function (half of duration of 
        # precipitation event/delivery of irrigation) 
        p_b = p_l1/2
        
        precip = Expression('p/(1 + exp(-100*(1 - pow(t - a1, 2)/pow(b, 2))))',
                            degree = 0, t = tt, a1 = p_a1, b = p_b, p = p)
        
    elif p_pat == 2:
        
        # Numerator of precipitation function (rate of precipitation)
        p = p_tot/(p_l1*p_lred*2)
        
        # Length of each precipitation/irrigation event
        p_l2 = p_l1*p_lred
        
        # Centre of first bump of precipitation function (halfway point of 
        # first precipitation event/delivery of irrigation)
        p_a21 = p_l2/2
        
        # Centre of second bump of precipitation function (halfway point of 
        # second precipitation event/delivery of irrigation)
        p_a22 = T - 1 + p_l2/2
        
        # Radii of bumps of precipitation function (half of duration of 
        # precipitation event/delivery of irrigation) 
        p_b = p_l2/2
        
        precip = Expression('p/(1 + exp(-100*(1 - pow(t - a1, 2)/pow(b, 2)))) + p/(1 + exp(-100*(1 - pow(t - a2, 2)/pow(b, 2))))',
                            degree = 0, t = tt, a1 = p_a21, a2 = p_a22, b = p_b, p = p)
        
    elif p_pat == 3:
        
        # Numerator of precipitation function (rate of precipitation)
        p = p_tot/(p_l1*(p_lred**2)*3)
        
        # Length of each precipitation/irrigation event
        p_l3 = p_l1*p_lred**2
        
        # Centre of first bump of precipitation function (halfway point of 
        # first precipitation event/delivery of irrigation)
        p_a31 = p_l3/2
        
        # Centre of second bump of precipitation function (halfway point of 
        # second precipitation event/delivery of irrigation)
        p_a32 = (T - 1)/2 + p_l3/2
        
        # Centre of third bump of precipitation function (halfway point of 
        # third precipitation event/delivery of irrigation)
        p_a33 = T - 1 + p_l3/2
        
        # Radii of bumps of precipitation function (half of duration of 
        # precipitation event/delivery of irrigation) 
        p_b = p_l3/2
        
        precip = Expression('p/(1 + exp(-100*(1 - pow(t - a1, 2)/pow(b, 2)))) + p/(1 + exp(-100*(1 - pow(t - a2, 2)/pow(b, 2)))) + p/(1 + exp(-100*(1 - pow(t - a3, 2)/pow(b, 2))))',
                            degree = 0, t = tt, a1 = p_a31, a2 = p_a32, a3 = p_a33, b = p_b, p = p)
    
    ###########################################################################        
    # Defining indicator function to calculate water content of the rooted 
    # zone
    ###########################################################################
    
    chi_root_zone = Expression('x[0] >= rd ? 1.0 : 0.0', degree = 0, rd = rooting_depth)
    
    chi_uptake_zone = Expression('x[0] <= uzs && x[0] >= uze ? 1.0 : 0.0', 
                                 degree = 0, uze = uptake_zone_end, uzs = uptake_zone_start)
    
    print('Vertex values of indicator function for rooted zone', chi_root_zone.compute_vertex_values(mesh))
    print('Vertex values of indicator function for uptake zone', chi_uptake_zone.compute_vertex_values(mesh))
    print('Vertices', mesh.coordinates()) 
    
    ###########################################################################
    # Setting some other required parameter values for implementation of the
    # model.
    ###########################################################################
    
    # Setting number of quadrature points in finite element computation.
    parameters["form_compiler"]["quadrature_degree"] = 7
    
    # Step size. 
    tau = T/Nt
    print('Time step size from function inputs =', tau)
    
    # Setting initial time.
    t = 0
    
    # Normal to the domain surface.
    normal = FacetNormal(mesh)
    
    # Canonical vector in vertical direction.
    e3 = as_vector([1])
    
    # Number of L-scheme iterations in the water transport solver.
    J_L = 4
    
    # Number of Newton iterations in the water transport solver.
    J_N = 16
    
    ###########################################################################
    # Defining the initial values
    ###########################################################################
    
    print('Initial water content when rhizodeposits not considered =', theta_init)
    
    if rhizodeposits == True:
        if cw_init == 'eq' and cd_init == 'eq':
            cd_init = kappa_d/(rho*(kappa_d + kappa_w))
            cw_init = (1/theta_init)*(ex_total - kappa_d/(kappa_d + kappa_w))
            print('rhizodeposit total at equilibrium', theta_init*cw_init + rho*cd_init)
        
        # Suspended rhizodeposit concentration
        cw0 = Expression('cw_f + ((cw_r - cw_f)/2)*(1 + (exp(a*(x[0] - rd)) - 1)/(exp(a*(x[0] - rd)) + 1))',
                          degree = 2, cw_f = 0.0, cw_r = cw_init, a = 1.0, rd = minx)
        # Dried rhizodeposit concentration.
        cd0 = Expression('cd_f + ((cd_r - cd_f)/2)*(1 + (exp(a*(x[0] - rd)) - 1)/(exp(a*(x[0] - rd)) + 1))',
                          degree = 2, cd_f = 0.0, cd_r = cd_init, a = 1.0, rd = minx)
        
    else:
        # Suspended rhizodeposit concentration
        cw0 = Expression('cw_f + ((cw_r - cw_f)/2)*(1 + (exp(a*(x[0] - rd)) - 1)/(exp(a*(x[0] - rd)) + 1))',
                          degree = 2, cw_f = 0.0, cw_r = 0.0, a = 1.0, rd = minx) 
        # Drien rhizodeposit concentration
        cd0 = Expression('cd_f + ((cd_r - cd_f)/2)*(1 + (exp(a*(x[0] - rd)) - 1)/(exp(a*(x[0] - rd)) + 1))',
                          degree = 2, cd_f = 0.0, cd_r = 0.0, a = 1.0, rd = minx)
        
    # Writing label for the sum of the product of initial water content and
    # suspended rhizodeposit concentration with the product of bulk density and 
    # initial attached rhizodeposit concentration.
    if type(cw_init) == float and type(cd_init) == float:
        ex_total = theta_init*cw_init + rho*cd_init
        print('rhizodeposit total not at equilibrium', ex_total)    
        
    ex_total_int = str(math.modf(ex_total)[1])[:-2]
    ex_total_dec = str(np.round(math.modf(ex_total)[0], 2))[2:]
    print('Integer component of rhizodeposit total', ex_total_int)
    print('Decimal component of rhizodeposit total', ex_total_dec)    
    print('rhizodeposit total', ex_total)
    
    # Initial condition of numerical solution to suspended rhizodeposit equation.
    cw_ = interpolate(cw0, W)
    # print('Initial suspended rhizodeposit concentration', cw_.vector()[:])
    
    # Initial condition of numerical solution to dried rhizodeposit equation.
    cd_ = interpolate(cd0, W)
    # print('Initial dried rhizodeposit concentration', cd_.vector()[:])
    
    # Setting the wetting/drying status for step before the initial  
    # condition.
    if status0 == 'wetting':
        status_j_2 = alpha_vg_w*np.ones(N_scalar_nodes)
    
        # Setting dummy initial pressure head
        h_init = inv_theta_hyst(theta_init, alpha_vg_w, theta_r0, theta_s0, n_vg, m_vg)
        h0 = Constant(h_init)
        theta0_test = theta_r0 + (theta_s0 - theta_r0)*se(h_init, alpha_vg_w, n_vg, m_vg)
        print('theta0 from h0 when not considering rhizodeposits =', theta0_test)
    
    else:
        status_j_2 = alpha_vg_d*np.ones(N_scalar_nodes)
        
        # Setting dummy initial pressure head
        h_init = inv_theta_hyst(theta_init, alpha_vg_d, theta_r0, theta_s0, n_vg, m_vg)
        h0 = Constant(h_init)
        theta0_test = theta_r0 + (theta_s0 - theta_r0)*se(h_init, alpha_vg_d, n_vg, m_vg)
        print('theta0 from h0 when not considering rhizodeposits =', theta0_test)
        
    # print('Wetting and drying status before initial condition', status_j_2)
    
    # Setting the pressure head condition for the step before that of the
    # initial condition to the same as the initial condition.
    hj_2 = interpolate(h0, W)
    # print('Pressure head before the initial condition =', hj_2.vector()[:])
    # print('Pressure head at initial condition =', h_.vector()[:])    
    
    # Setting the alpha_vg parameter for step before the initial
    # condition.
    alpha_vg_j_2 = alpha_vg_func(status_j_2, hj_2, alpha_vg_d, alpha_vg_w, a_st, b_st, a_ca, b_ca, cw_, cd_, W)
    print('Inverse of air entry pressure head before initial condition', alpha_vg_j_2.vector()[:])
    
    # Changing initial pressure head if rhizodeposits included so that initial water
    # content is the same as without rhizodeposits
    if rhizodeposits == True:
        # Applying inverse of water retention function, with alpha value altered
        # by rhizodeposits, to initial water content of simulation without rhizodeposits. 
        h_init_ex = inv_theta_hyst(theta_init, alpha_vg_j_2.vector()[0], theta_r0, theta_s0, n_vg, m_vg)
        print('h0 in rooted section of domain when considering rhizodeposits =', h_init_ex)
        
        theta0_test = theta_r0 + (theta_s0 - theta_r0)*se(h_init_ex, alpha_vg_j_2.vector()[0], n_vg, m_vg)
        print('theta0 in rooted section of domain when considering rhizodeposits =', theta0_test)
        
        h0 = Expression('h_f + ((h_r - h_f)/2)*(1 + (exp(a*(x[0] - rd)) - 1)/(exp(a*(x[0] - rd)) + 1))',
                          degree = 2, h_f = h_init, h_r = h_init_ex, a = 1.0, rd = minx)
    
    # Setting the pressure head condition for the step before that of the
    # initial condition to the same as the initial condition.
    hj_2 = interpolate(h0, W)
    print('Pressure head before the initial condition =', hj_2.vector()[:])
    
    # Initial condition of numerical solution to pressure equation.
    h_ = interpolate(h0, W)
    print('Initial pressure head =', h_.vector()[:])
    
    # Setting initial value for residual water content for the step
    # before the initial condition.
    theta_r_j_2_exp = Constant(theta_r0)
    theta_r_j_2 = interpolate(theta_r_j_2_exp, W)
    
    # Setting initial value for saturated water content two steps ago.
    theta_s_j_2_exp = Constant(theta_s0)
    theta_s_j_2 = interpolate(theta_s_j_2_exp, W)
    
    # Setting initial function for water content from 2 steps ago.
    theta_j_2 = vec_theta_hyst(hj_2, alpha_vg_j_2, theta_r_j_2, theta_s_j_2, n_vg, m_vg, W)
    print('Water content before initial condition', theta_j_2.vector()[:])
    
    # Setting initial value for visualisation water content from previous 
    # timestep.
    theta_vis_2 = vec_theta_hyst(hj_2, alpha_vg_j_2, theta_r_j_2, theta_s_j_2, n_vg, m_vg, W)
    
    # Setting initial function for saturated hydraulic conductivity from 2
    # steps ago.
    Ks_j_3 = Function(W)
    
    if status0 == 'wetting':
        Ks_j_3.vector()[:] = np.ones(N_scalar_nodes)*Ksw
    else:
        Ks_j_3.vector()[:] = np.ones(N_scalar_nodes)*Ksd
    
    Ks_j_2 = Ks_func(status_j_2, hj_2, hj_2, Ks_j_3, alpha_vg_d, Ksd, Ksw, a_st, b_st, a_ca, b_ca, a_vis, b_vis, cw_, cd_, beta, W)
    # print('Saturated hydraulic conductivity before the initial condition', Ks_j_2.vector()[:])
    
    # Setting initial Kstar function for 2 steps ago.
    Kstar_j_2 = Function(W)
    Kstar_j_2.vector()[:] = np.zeros(N_scalar_nodes)
    
    # Setting initial value for hysteresis parameter 2 from 2 steps ago
    # in the regularised hysteretic hydraulic conductivity function.
    alphaK_j_2 = Function(W)
    alphaK_j_2.vector()[:] = np.ones(N_scalar_nodes)
    
    # Setting hydraulic conductivity from 2 steps ago.
    K_j_2 = vec_K_hyst(hj_2, Ks_j_2, Kstar_j_2, alphaK_j_2, alpha_vg_j_2, n_vg, m_vg, theta_r0, theta_s0, W)
    # print('Hydraulic conductivity before the initial condition', K_j_2.vector()[:])
    
    # Setting wetting/drying status at initial condition.
    status_j_ = wet_dry_status(h_.vector()[:], hj_2.vector()[:], status_j_2, alpha_vg_d, alpha_vg_w, tol)
    # print('Initial status', status_j_)
    
    # Functional representation of status to visualise its value.
    status_vis_ = Function(W)
    status_vis_.vector()[:] = status_j_
    
    # Setting value of air entry pressure head at initial condition.
    alpha_vg_j_ = alpha_vg_func(status_j_, h_, alpha_vg_d, alpha_vg_w, a_st, b_st, a_ca, b_ca, cw_, cd_, W)
    # print('Initial alpha values', alpha_vg_j_.vector()[:])
    
    # Functional representation of alpha_vg to visualise its value.
    alpha_vg_vis_ = Function(W)
    alpha_vg_vis_.vector()[:] = alpha_vg_j_.vector()[:]
    
    # Setting the value for the residual water content in the
    # hysteretic water content function at the initial condition.
    theta_r_j_ = theta_r_func(status_j_, h_, hj_2, theta_j_2, alpha_vg_j_, alpha_vg_w, theta_r0, theta_s0, n_vg, m_vg, W)
    # print('Initial residual water content values', theta_r_j_.vector()[:])
    
    # Setting the initial value for the saturated water content in the 
    # hysteretic water content function.
    theta_s_j_ = theta_s_func(status_j_, h_, hj_2, theta_j_2, alpha_vg_j_, alpha_vg_d, theta_r0, theta_s0, n_vg, m_vg, W)
    # print('Initial saturated water content values', theta_s_j_.vector()[:])
    
    # Setting initial value for saturated hydraulic conductivity values from 
    # previous timestep.
    Ks_j_ = Ks_func(status_j_, h_, hj_2, Ks_j_2, alpha_vg_d, Ksd, Ksw, a_st, b_st, a_ca, b_ca, a_vis, b_vis, cw_, cd_, beta, W)
    # print('Initial saturated conductivity', Ks_j_.vector()[:])
    
    # Setting initial value for hysteresis parameter 1 from 1 step ago
    # in the regularised hysteretic hydraulic conductivity function.
    Kstar_j_ = Kstar_func(h_, hj_2, K_j_2, Ks_j_, alpha_vg_j_, alpha_vg_d, Kr_max, status_j_, n_vg, m_vg, theta_r0, theta_s0, W)
    # print('Initial conductivity hysteresis parameter 1', Kstar_j_.vector()[:])
    
    # Setting initial value for hysteresis parameter 2 from 1 step ago
    # in the regularised hysteretic hydraulic conductivity function.
    alphaK_j_ = alphaK_func(h_, hj_2, K_j_2, Ks_j_, alpha_vg_j_, alpha_vg_d, Kr_max, status_j_, n_vg, m_vg, theta_r0, theta_s0, W)
    # print('Initial conductivity hysteresis parameter 2', alphaK_j_.vector()[:])
    
    # Setting initial value for visualisation hydraulic conductivity from 
    # previous timestep.
    K_vis_ = vec_K_hyst(h_, Ks_j_, Kstar_j_, alphaK_j_, alpha_vg_j_, n_vg, m_vg, theta_r0, theta_s0, W)
    K_vis_j_ = vec_K_hyst(h_, Ks_j_, Kstar_j_, alphaK_j_, alpha_vg_j_, n_vg, m_vg, theta_r0, theta_s0, W)
    # print('Initial hydraulic conductivity', K_vis_j_.vector()[:])
    
    # Setting initial value for visualisation water content from previous 
    # timestep.
    theta_vis_ = vec_theta_hyst(h_, alpha_vg_j_, theta_r_j_, theta_s_j_, n_vg, m_vg, W)
    theta_vis_j_ = vec_theta_hyst(h_, alpha_vg_j_, theta_r_j_, theta_s_j_, n_vg, m_vg, W)
    print('Initial water content', theta_vis_j_.vector()[:])
    
    # Setting initial value for stress function in root water uptake.
    alpha_f_j_ = alpha_f_func(theta_vis_j_, theta1, theta2, theta3, theta3, W)
    
    # Setting initial value for compensation term in root water uptake.
    comp_j_ = compensation(theta_vis_j_, theta1, theta2, theta3, theta4, NRLD, W, dx)
    
    # Setting initial value of evaporation coefficient for boundary condition.
    Ke_j_ = Ke_func(theta_vis_j_, Kcmax, Kcb, theta_fc, theta_wp, W)
    
    # Setting initial value for effective saturation.
    se_ = vec_se_hyst(h_, alpha_vg_j_, n_vg, m_vg, W)
    # print('Initial effective saturation', se_.vector()[:])
    
    # Setting initial value for water flux.
    q_2 = q(K_vis_, h_, V, e3)
    q_ = q(K_vis_, h_, V, e3)
    # print('Initial water flux', q_.vector()[:])
    
    # Setting initial value for L in the L-scheme.
    L_theta_nodes_j_ = L_retention(alpha_vg_j_.vector()[:], theta_r_j_.vector()[:], theta_s_j_.vector()[:],  n_vg, -500, -eps)        
    # print("Initial tight upper bound for L at each node =", L_theta_nodes_j_)
    
    L_theta = np.max(L_theta_nodes_j_)
    # print('Initial L value to be used in numerical scheme =', L_theta)
    
    # Setting initial evaporation.
    evaporation_ = vec_evaporation(theta_vis_, W,  Kcmax, Kcb, theta_fc, theta_wp)(tp)
    # print('Initial evaporation', evaporation_)
    
    # Setting initial free drainage.
    free_drainage_ = K_vis_(bttm)
    # print('Initial free drainage', free_drainage_)
    
    # Setting initial flux at boundary between vegetated and fallow soil.
    deep_percolation_ = q_(rooting_depth)
    # print('Initial deep_percolation', deep_percolation_)
    
    # Setting initial flux at boundary where the uptake zone starts
    upper_flux_ = q_(uptake_zone_start)
    # print('Initial upper_flux', upper_flux_)
    
    # Setting initial flux at boundary where the uptake zone ends
    lower_flux_ = q_(uptake_zone_end)
    # print('Initial lower_flux', lower_flux_)
    
    # Setting initial precipiation.
    precip_ = precip(tp)
    # print('Initial precipitation', precip_)
    
    # Setting initial water loss conditions.
    runoff_ = runoff(theta_vis_, theta_s0, precip_, 250)

    # runoff_float_ = runoff(theta_vis_(tp), theta_s0, precip_, 250)
    # print('Initial runoff', runoff_float_)
    
    # Setting initial uptake.
    S_ = vec_S(W, theta_vis_, theta1, theta2, theta3, theta4, Tp, NRLD, dx)
    # print('Initial uptake', S_.vector()[:])   
    
    ###########################################################################
    # Recording visualisations of initial conditions.
    ###########################################################################
    
    # xdmffile_h = XDMFFile(f'data/h{name}_{ex_lab}T{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}_extot{ex_total_int}_{ex_total_dec}_{imported_mesh}.xdmf')
    # xdmffile_theta = XDMFFile(f'data/th{name}_{ex_lab}T{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}_{imported_mesh}.xdmf')
    # xdmffile_K = XDMFFile(f'data/K{name}_{ex_lab}T{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}_{imported_mesh}.xdmf')
    # xdmffile_alpha = XDMFFile(f'data/alp{name}_{ex_lab}T{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}_{imported_mesh}.xdmf')
    # xdmffile_status = XDMFFile(f'data/stat{name}_{ex_lab}T{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}_{imported_mesh}.xdmf')
    # xdmffile_se = XDMFFile(f'data/se{name}_{ex_lab}T{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}_{imported_mesh}.xdmf')
    # xdmffile_q = XDMFFile(f'data/q{name}_{ex_lab}T{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}_{imported_mesh}.xdmf')
    # xdmffile_cw = XDMFFile(f'data/cw{name}_{ex_lab}T{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}_{imported_mesh}.xdmf')
    # xdmffile_cd = XDMFFile(f'data/cd{name}_{ex_lab}T{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}_{imported_mesh}.xdmf')
        
    # if rhizodeposits != False:
        # xdmffile_h.write(h_, t)
        # xdmffile_theta.write(theta_vis_, t)
        # xdmffile_alpha.write(alpha_vg_vis_,t)
        # xdmffile_status.write(status_vis_,t)
        # xdmffile_K.write(K_vis_, t)
        # xdmffile_se.write(se_, t)
        # xdmffile_q.write(q_, t)
        # xdmffile_cw.write(cw_, t)
        # xdmffile_cd.write(cd_, t)
        
    # else:
        # xdmffile_h.write(h_, t)
        # xdmffile_theta.write(theta_vis_, t)
        # xdmffile_alpha.write(alpha_vg_vis_,t)
        # xdmffile_status.write(status_vis_,t)
        # xdmffile_K.write(K_vis_, t)
        # xdmffile_se.write(se_, t)
        # xdmffile_q.write(q_, t)
        
    ###########################################################################
    # Defining variational form for soil water transport
    ###########################################################################
    
    # Trial function.
    hj = TrialFunction(W)
    
    # Test function.
    phi_h = TestFunction(W)
    
    # Function for solution at previous linearisation iteration.
    hj_ = interpolate(h0, W)
    
    # L_scheme bilinear form.
    a_L = L_theta*hj*phi_h*dx + tau*dot(K_hyst(hj_, Ks_j_, Kstar_j_, alphaK_j_, alpha_vg_j_, n_vg, m_vg, theta_r0, theta_s0)*grad(hj), grad(phi_h))*dx
    
    # L_scheme linear form.
    L_L = - tau*dot(K_hyst(hj_, Ks_j_, Kstar_j_, alphaK_j_, alpha_vg_j_, n_vg, m_vg, theta_r0, theta_s0)*e3, grad(phi_h))*dx \
          + theta_hyst(h_, alpha_vg_j_, theta_r_j_, theta_s_j_, n_vg, m_vg)*phi_h*dx \
          - theta_hyst(hj_, alpha_vg_j_, theta_r_j_, theta_s_j_, n_vg, m_vg)*phi_h*dx \
          + L_theta*hj_*phi_h*dx - tau*S(Tp, comp_j_, alpha_f_j_, NRLD)*phi_h*dx \
          - tau*(evaporation(Ke_j_) + runoff(theta_hyst(hj_, alpha_vg_j_, theta_r_j_, theta_s_j_, n_vg, m_vg), theta_s0, precip, 250) - precip)*phi_h*ds(2) \
          - tau*(K_hyst(hj_, Ks_j_, Kstar_j_, alphaK_j_, alpha_vg_j_, n_vg, m_vg, theta_r0, theta_s0))*phi_h*ds(1)    
    
    # Newton bilinear form.
    a_N = theta_prime_newt(hj_, alpha_vg_j_, theta_r_j_, theta_s_j_, n_vg)*hj*phi_h*dx \
          + tau*dot(K_hyst(hj_, Ks_j_, Kstar_j_, alphaK_j_, alpha_vg_j_, n_vg, m_vg, theta_r0, theta_s0)*grad(hj), grad(phi_h))*dx
    
    # Newton linear form.
    L_N = - tau*dot(K_hyst(hj_, Ks_j_, Kstar_j_, alphaK_j_, alpha_vg_j_, n_vg, m_vg, theta_r0, theta_s0)*e3, grad(phi_h))*dx \
          + theta_hyst(h_, alpha_vg_j_, theta_r_j_, theta_s_j_, n_vg, m_vg)*phi_h*dx \
          - theta_hyst(hj_, alpha_vg_j_, theta_r_j_, theta_s_j_, n_vg, m_vg)*phi_h*dx \
          + theta_prime_newt(hj_, alpha_vg_j_, theta_r_j_, theta_s_j_, n_vg)*hj_*phi_h*dx \
          - tau*S(Tp, comp_j_, alpha_f_j_, NRLD)*phi_h*dx \
          - tau*(evaporation(Ke_j_) + runoff(theta_hyst(hj_, alpha_vg_j_, theta_r_j_, theta_s_j_, n_vg, m_vg), theta_s0, precip, 250) - precip)*phi_h*ds(2) \
          - tau*(K_hyst(hj_, Ks_j_, Kstar_j_, alphaK_j_, alpha_vg_j_, n_vg, m_vg, theta_r0, theta_s0))*phi_h*ds(1)
          
    ###########################################################################
    # Defining variational form for rhizodeposit transport and conversion between
    # wet and dry rhizodeposits.
    ###########################################################################      
    
    # Trial function.
    cw = TrialFunction(W)
    
    # Test function.
    phi_cw = TestFunction(W)

    ################
    # Alternative model for rhizodeposit transport semi-inspired by Simunek et 
    # al,.(2006) "Colloid-transport ..." Both the wetting and drying of
    # rhizodeposits is incorporated.
    
    # Better reference for the model would be Schnepf, Leithner, Klepsch 2012
    # "Modeling phosphorus uptake by a growing and exuding root system" 
    
    # Bilinear form
    a_cw = theta_vis_*cw*phi_cw*dx \
            + tau*dot(D_cw*theta_vis_*grad(cw), grad(phi_cw))*dx \
            - tau*dot(q_*cw, grad(phi_cw))*dx \
            + tau*kappa_d*theta_vis_*cw*phi_cw*dx

    # # Linear form
    L_cw = tau*rho*kappa_w*cd_*phi_cw*dx + theta_vis_2*cw_*phi_cw*dx \
            + tau*r_cw*RSA*theta_vis_*phi_cw*dx
            
    ###########################################################################
    # Implementing numerical schemes to obtain pressure head and rhizodeposit
    # concentrations.
    ###########################################################################
    
    # Initiating solutions.
    hj = Function(W)
    cw = Function(W)
    cd = Function(W)
    
    # Looping over all time steps
    for n in range(Nt):
        
        # Updating timestep.
        t += tau
        
        tt.assign(t)
        
        precip_ = precip(tp)
        # print('precipitation at t =', t, 'is precip =', precip_)
        
        # Looping over all L-scheme linearisation iterations.
        for j_L in range(J_L):
            
            # Assembling bilinear form.
            A_L = assemble(a_L)
            
            # Assembling linear form.
            b_L = assemble(L_L)
            
            # Solving the problem.
            solve(A_L, hj.vector(), b_L)
            
            # Updating h^{n, j-2} with head solution at current linearisation 
            # iteration.
            hj_2.assign(hj_)
            
            # Updating water content from two steps ago
            theta_j_2.vector()[:] = theta_vis_j_.vector()[:]
            
            # Updating hydraulic conductivity from two steps ago
            K_j_2.vector()[:] = K_vis_j_.vector()[:]
            
            # Updating saturated hydraulic conductivity from two steps ago
            Ks_j_2.assign(Ks_j_)
            
            # Updating the wetting/drying status from two steps ago
            status_j_2 = status_j_
            
            # Updating h^{n, j-1} with head solution at current linearisation 
            # iteration.
            hj_.assign(hj)
            # print('Nodal values of pressure head at t =', t, 'and L-scheme iteration', j_L, 'are ', hj_.vector()[:])
            
            # Updating wetting/drying status.
            status_j_ = wet_dry_status(hj_.vector()[:], hj_2.vector()[:], status_j_2, alpha_vg_d, alpha_vg_w, tol)
            
            # Updating the value of the alpha_vg_parameter.
            alpha_vg_j_.assign(alpha_vg_func(status_j_, hj_, alpha_vg_d, alpha_vg_w, a_st, b_st, a_ca, b_ca, cw_, cd_, W))
            # print('Inverse air entry pressure head at t =', t, 'and L-scheme iteration', j_L, 'is', alpha_vg_j_.vector()[:])
            
            # Updating the value of the residual water content in the
            # hysteretic water content function.
            theta_r_j_.assign(theta_r_func(status_j_, hj_, hj_2, theta_j_2, alpha_vg_j_, alpha_vg_w, theta_r0, theta_s0, n_vg, m_vg, W))
            # print('Residual water content at t =', t, 'and L-scheme iteration', j_L, 'is', theta_r_j_.vector()[:])
            
            # Updating the value for the saturated water content in the 
            # hysteretic water content function.
            theta_s_j_.assign(theta_s_func(status_j_, hj_, hj_2, theta_j_2, alpha_vg_j_, alpha_vg_d, theta_r0, theta_s0, n_vg, m_vg, W))
            # print('Saturated water content at t =', t, 'and L-scheme iteration', j_L, 'is', theta_s_j_.vector()[:])
            
            # Updating the value for saturated hydraulic conductivity values. 
            Ks_j_.assign(Ks_func(status_j_, hj_, hj_2, Ks_j_2, alpha_vg_d, Ksd, Ksw, a_st, b_st, a_ca, b_ca, a_vis, b_vis, cw_, cd_, beta, W))
            # print('j Saturated hydraulic conductivity at t =', t, 'and L-scheme iteration', j_L, 'is', Ks_j_.vector()[:])
            
            # Updating the value for hysteresis parameter 1 from 1 time step ago
            # in the regularised hysteretic hydraulic conductivity function.
            Kstar_j_.assign(Kstar_func(hj_, hj_2, K_j_2, Ks_j_, alpha_vg_j_, alpha_vg_d, Kr_max, status_j_, n_vg, m_vg, theta_r0, theta_s0, W))
            # print('Kstar at t =', t, 'and L-scheme iteration', j_L, 'is', Kstar_j_.vector()[:])
            
            # Updating the value for hysteresis parameter 2 from 1 time step ago
            # in the regularised hysteretic hydraulic conductivity function.
            alphaK_j_.assign(alphaK_func(hj_, hj_2, K_j_2, Ks_j_, alpha_vg_j_, alpha_vg_d, Kr_max, status_j_, n_vg, m_vg, theta_r0, theta_s0, W))
            # print('j alphaK at t =', t, 'and L-scheme iteration', j_L, 'is', alphaK_j_.vector()[:])
            
            # Updating the value for hydraulic conductivity from previous timestep
            K_vis_j_.vector()[:] = vec_K_hyst(hj_, Ks_j_, Kstar_j_, alphaK_j_, alpha_vg_j_, n_vg, m_vg, theta_r0, theta_s0, W).vector()[:]
            # print('Nodal values of hydraulic conductivity at t =', t, 'and L-scheme iteration', j_L, 'are ', K_vis_.vector()[:])
            
            # Updating the value for hysteretic soil water content.
            theta_vis_j_.vector()[:] = vec_theta_hyst(hj_, alpha_vg_j_, theta_r_j_, theta_s_j_, n_vg, m_vg, W).vector()[:]
            # print('Nodal values of water content at t =', t, 'and L-scheme iteration', j_L, 'are ', theta_vis_.vector()[:])
            
            # Updating value for stress function in root water uptake.
            alpha_f_j_ = alpha_f_func(theta_vis_j_, theta1, theta2, theta3, theta3, W)
            
            # Updating value for compensation term in root water uptake.
            comp_j_ = compensation(theta_vis_j_, theta1, theta2, theta3, theta4, NRLD, W, dx)
            
            # Setting initial value of evaporation coefficient for boundary condition.
            Ke_j_ = Ke_func(theta_vis_j_, Kcmax, Kcb, theta_fc, theta_wp, W)
            
            # Updating value for L in the L-scheme.
            L_theta_nodes_j_ = L_retention(alpha_vg_j_.vector()[:], theta_r_j_.vector()[:], theta_s_j_.vector()[:], n_vg, -500, -eps)        
            # print("Current tight upper bound for L at each node =", L_theta_nodes_)
            
            L_theta = np.max(L_theta_nodes_j_)
            # print('Current L value to be used in numerical scheme =', L_theta)
            
        # Looping over all Newton method iterations.
        for j_N in range(J_N):
            
            # Assembling bilinear form
            A_N = assemble(a_N)
            
            # Assembling linear form
            b_N = assemble(L_N)
            
            # Solve the problem.
            solve(A_N, hj.vector(), b_N)
            
            # Updating h^{n, j-2} with head solution at current linearisation 
            # iteration.
            hj_2.assign(hj_)
            
            # Updating water content from two steps ago
            theta_j_2.vector()[:] = theta_vis_j_.vector()[:]
            
            # Updating hydraulic conductivity from two steps ago
            K_j_2.vector()[:] = K_vis_j_.vector()[:]
            
            # Updating saturated hydraulic conductivity from two steps ago
            Ks_j_2.assign(Ks_j_)
            
            # Updating the wetting/drying status from two steps ago
            status_j_2 = status_j_
            
            # Updating h^{n, j-1} with head solution at current linearisation 
            # iteration.
            hj_.assign(hj)
            # print('Nodal values of pressure head at t =', t, 'and Newton iteration', j_N, 'are ', hj_.vector()[:])
            
            # Updating wetting/drying status.
            status_j_ = wet_dry_status(hj_.vector()[:], hj_2.vector()[:], status_j_2, alpha_vg_d, alpha_vg_w, tol)
            
            # Updating the value of the alpha_vg_parameter.
            alpha_vg_j_.assign(alpha_vg_func(status_j_, hj_, alpha_vg_d, alpha_vg_w, a_st, b_st, a_ca, b_ca, cw_, cd_, W))
            # print('Inverse air entry pressure head at t =', t, 'and Newton iteration', j_N, 'is', alpha_vg_j_.vector()[:])
            
            # Updating the value of the residual water content in the
            # hysteretic water content function.
            theta_r_j_.assign(theta_r_func(status_j_, hj_, hj_2, theta_j_2, alpha_vg_j_, alpha_vg_w, theta_r0, theta_s0, n_vg, m_vg, W))
            # print('Residual water content at t =', t, 'and Newton iteration', j_N, 'is', theta_r_j_.vector()[:])
            
            # Updating the value for the saturated water content in the 
            # hysteretic water content function.
            theta_s_j_.assign(theta_s_func(status_j_, hj_, hj_2, theta_j_2, alpha_vg_j_, alpha_vg_d, theta_r0, theta_s0, n_vg, m_vg, W))
            # print('Saturated water content at t =', t, 'and Newton iteration', j_N, 'is', theta_s_j_.vector()[:])
            
            # Updating the value for saturated hydraulic conductivity values. 
            Ks_j_.assign(Ks_func(status_j_, hj_, hj_2, Ks_j_2, alpha_vg_d, Ksd, Ksw, a_st, b_st, a_ca, b_ca, a_vis, b_vis, cw_, cd_, beta, W))
            # print('Saturated hydraulic conductivity at time', t, 'and Newton iteration', j_N, ' is', Ks_j_.vector()[:])
            
            # Updating the value for hysteresis parameter 1 from 1 time step ago
            # in the regularised hysteretic hydraulic conductivity function.
            Kstar_j_.assign(Kstar_func(hj_, hj_2, K_j_2, Ks_j_, alpha_vg_j_, alpha_vg_d, Kr_max, status_j_, n_vg, m_vg, theta_r0, theta_s0, W))
            # print('Kstar at t =', t, 'and Newton iteration', j_N, 'is', Kstar_j_.vector()[:])
            
            # Updating the value for hysteresis parameter 2 from 1 time step ago
            # in the regularised hysteretic hydraulic conductivity function.
            alphaK_j_.assign(alphaK_func(hj_, hj_2, K_j_2, Ks_j_, alpha_vg_j_, alpha_vg_d, Kr_max, status_j_, n_vg, m_vg, theta_r0, theta_s0, W))
            # print('alphaK at t =', t, 'and Newton iteration', j_N, 'is', alphaK_j_.vector()[:])
            
            # Updating the value for hydraulic conductivity from previous timestep
            K_vis_j_.vector()[:] = vec_K_hyst(hj_, Ks_j_, Kstar_j_, alphaK_j_, alpha_vg_j_, n_vg, m_vg, theta_r0, theta_s0, W).vector()[:]
            # print('Nodal values of hydraulic conductivity at t =', t, 'and Newton iteration', j_N, 'are ', K_vis_j_.vector()[:])
            
            # Updating the value for hysteretic soil water content.
            theta_vis_j_.vector()[:] = vec_theta_hyst(hj_, alpha_vg_j_, theta_r_j_, theta_s_j_, n_vg, m_vg, W).vector()[:]
            # print('Nodal values of water content at t =', t, 'and Newton iteration', j_N, 'are ', theta_vis_j_.vector()[:])
            
            # Updating value for stress function in root water uptake.
            alpha_f_j_ = alpha_f_func(theta_vis_j_, theta1, theta2, theta3, theta3, W)
            
            # Updating value for compensation term in root water uptake.
            comp_j_ = compensation(theta_vis_j_, theta1, theta2, theta3, theta4, NRLD, W, dx)
            
            # Setting initial value of evaporation coefficient for boundary condition.
            Ke_j_ = Ke_func(theta_vis_j_, Kcmax, Kcb, theta_fc, theta_wp, W)
            
        # Updating h^{n-1} to the head solution at current timestep after all 
        # linearisation iterations have been performed.
        h_.assign(hj)
        # print('Nodal values of pressure head at t =', t, 'pattern =', p_pat, 'are', h_.vector()[:])
        
        # Updating function for visualisation of hydraulic conductivity.
        K_vis_.vector()[:] = K_vis_j_.vector()[:]
        # print('Nodal values of hydraulic conductivity at t =', t, 'are ', K_vis_.vector()[:])
        
        # Updating function for visualisation of water content at previous timestep.
        theta_vis_2.assign(theta_vis_)
        
        # Updating function for visualisation of water content at previous timestep.
        theta_vis_.assign(theta_vis_j_) 
        print('Nodal values of water content at t =', t, 'pattern =', p_pat, 'are', theta_vis_.vector()[:])
        
        # Updating the function for the flux at previous timestep.
        q_2.assign(q_)
        
        # Updating the function for the flux at this timestep.
        q_.assign(q(K_vis_, h_, V, e3))
        # print('Flux at boundary between vegetated and fallow soil at t =', t, 'pattern =', p_pat, 'is', q_(minx))
        # print('Flux at soil surface at t =', t, 'pattern =', p_pat, 'is', q_(tp))
        
        # Updating the visual representation of the alpha_vg_parameter.
        alpha_vg_vis_.vector()[:] = alpha_vg_j_.vector()[:]
        # print('Nodal values of inverse air entry pressure head at t =', t, 'are ', alpha_vg_vis_.vector()[:])
        # print('Maximum nodal value of inverse air entry pressure head at t =', t, 'is', np.max(alpha_vg_vis_.vector()[:]))
        # print('Minimum nodal value of inverse air entry pressure head at t =', t, 'is', np.min(alpha_vg_vis_.vector()[:]))
        
        # Updating visual representation of wetting and drying status. 
        status_vis_.vector()[:] = status_j_
        print('Wetting and drying status at t =', t, 'pattern =', p_pat, 'are', status_vis_.vector()[:])
        
        # Updating the functions for effective saturation 
        se_.assign(vec_se_hyst(h_, alpha_vg_j_, n_vg, m_vg, W))
        # print('Nodal values of effective saturation at t =', t, 'are', se_.vector()[:])
        
        ################
        # Solving suspended rhizodeposit variational form if they are present in the
        # soil.
        if rhizodeposits != False:
            # Assembling bilinear form of suspended rhizodeposit concentration problem.
            A_cw = assemble(a_cw)
            
            # Assembling linear form of suspended rhizodeposit concentration problem.
            b_cw = assemble(L_cw)
            
            # Solve the of suspended rhizodeposit concentration problem.
            solve(A_cw, cw.vector(), b_cw)
        
            # Updating the value of the suspended rhizodeposit concentration from the previous
            # timestep
            cw_.assign(cw)
            # print('Nodal values of suspended rhizodeposit concentration at t =', t, 'pattern =', p_pat, 'are', cw_.vector()[:])
            print('Minimum nodal value of suspended rhizodeposit at t =', t, 'is ', np.min(cw_.vector()[:]))
            print('Maximum nodal value of suspended rhizodeposit at t =', t, 'is ', np.max(cw_.vector()[:]))
            
            cd_comp = (tau*kappa_d*theta_vis_*cw_ + rho*cd_)/(rho*(1 + tau*kappa_w))
            
            cd = project(cd_comp, W)
            
            cd_.assign(cd)
            # print('Nodal values of dried rhizodeposit concentration at t =', t, 'pattern =', p_pat, 'are', cd_.vector()[:])
            print('Minimum nodal value of dried rhizodeposit at t =', t, 'is ', np.min(cd_.vector()[:]))
            print('Maximum nodal value of dried rhizodeposit at t =', t, 'is ', np.max(cd_.vector()[:]))
            
            # if np.min(cw_.vector()[:]) < 0 or np.min(cd_.vector()[:]) < 0:
                # raise ValueError('Exudation concentration cannot be negative')
            
            # Recording visualisations of at this timestep. 
            # xdmfile_cw.write(cw_, t)
            # xdmffile_cd.write(cd_, t)
            
        # Updating the value of the alpha_vg_parameter given the update in
        # suspended rhizodeposit concentration.
        alpha_vg_j_.assign(alpha_vg_func(status_j_, hj_, alpha_vg_d, alpha_vg_w, a_st, b_st, a_ca, b_ca, cw_, cd_, W))
        # print('Inverse air entry pressure head at t =', t, 'is', alpha_vg_j_.vector()[:])
        
        # Updating the visual representation of the alpha_vg_parameter.
        alpha_vg_vis_.vector()[:] = alpha_vg_j_.vector()[:]
        
        # Updating the value of the residual water content in the
        # hysteretic water content function given the new rhizodeposit 
        # concentration.
        theta_r_j_.assign(theta_r_func(status_j_, hj_, hj_2, theta_j_2, alpha_vg_j_, alpha_vg_w, theta_r0, theta_s0, n_vg, m_vg, W))
        # print('Residual water content at t =', t, 'is', theta_r_j_.vector()[:])
        
        # Updating the value of the aturated water content in the
        # hysteretic water content function given the new rhizodeposit 
        # concentration.
        theta_s_j_.assign(theta_s_func(status_j_, hj_, hj_2, theta_j_2, alpha_vg_j_, alpha_vg_d, theta_r0, theta_s0, n_vg, m_vg, W))
        # print('Saturated water content at t =', t, 'is', theta_s_j_.vector()[:])
        
        # Updating the value for saturated hydraulic conductivity values
        # given the update to rhizodeposit concentration.
        Ks_j_.assign(Ks_func(status_j_, hj_, hj_2, Ks_j_2, alpha_vg_d, Ksd, Ksw, a_st, b_st, a_ca, b_ca, a_vis, b_vis, cw_, cd_, beta, W))
        # print('Saturated hydraulic conductivity at time', t, ' is', Ks_j_.vector()[:])
        
        # Updating the value for hysteresis parameter 1 from 1 time step ago
        # in the regularised hysteretic hydraulic conductivity function.
        # Given the update to rhizodeposit concentration.
        Kstar_j_.assign(Kstar_func(hj_, hj_2, K_j_2, Ks_j_, alpha_vg_j_, alpha_vg_d, Kr_max, status_j_, n_vg, m_vg, theta_r0, theta_s0, W))
        # print('Kstar at t =', t, 'is', Kstar_j_.vector()[:])
        
        # Updating the value for hysteresis parameter 2 from 1 time step ago
        # in the regularised hysteretic hydraulic conductivity function.
        # Given the update to rhizodeposit concentration.
        alphaK_j_.assign(alphaK_func(hj_, hj_2, K_j_2, Ks_j_, alpha_vg_j_, alpha_vg_d, Kr_max, status_j_, n_vg, m_vg, theta_r0, theta_s0, W))
        # print('alphaK at t =', t, 'is', alphaK_j_.vector()[:])
        
        # Updating value for L in the L-scheme.
        # Given the update to rhizodeposit concentration.
        L_theta_nodes_j_ = L_retention(alpha_vg_j_.vector()[:], theta_r_j_.vector()[:], theta_s_j_.vector()[:], n_vg, -500, -eps)        
        # print("Current tight upper bound for L at each node =", L_theta_nodes_)
        
        L_theta = np.max(L_theta_nodes_j_)
        # print('Current L value to be used in numerical scheme =', L_theta)
        
        # Recording visualisations at this step.
        # xdmffile_h.write(h_, t)
        # xdmffile_theta.write(theta_vis_, t)
        # xdmffile_alpha.write(alpha_vg_vis_,t)
        # xdmffile_status.write(status_vis_,t)
        # xdmffile_K.write(K_vis_, t)
        # xdmffile_se.write(se_, t)
        # xdmffile_q.write(q_, t)
        
        # Computing post processing quantities before updating solution value    
        # Precipitation
        precip_ = precip(tp)
        total_precipitation[n] = precip_
        # print(f' Total precipitation = {p_tot_int}.{p_tot_dec}, event length = {p_l1_int}.{p_l1_dec}, reduction rate = {p_lred_int}.{p_lred_dec}, rhizodeposits = {ex_lab}: precipitation at t =', t, 'is', total_precipitation[n])
        print('Precipitation at t =', t, 'is', total_precipitation[n])
        
        runoff_float_ = runoff(theta_vis_(tp), theta_s0, precip_, 250)
        total_runoff[n] = runoff_float_
        print('Runoff at t =', t, 'is', total_runoff[n])
        
        root_zone_water_content[n] = assemble(theta_vis_*chi_root_zone*dx)
        print('Root zone water content at t =', t, 'is', root_zone_water_content[n])
        
        evaporation_ = vec_evaporation(theta_vis_, W,  Kcmax, Kcb, theta_fc, theta_wp)(tp)
        total_evaporation[n] = vec_evaporation(theta_vis_, W,  Kcmax, Kcb, theta_fc, theta_wp)(tp)
        print('evaporation at t =', t, 'is', total_evaporation[n])

        free_drainage_ = K_vis_(bttm)
        total_free_drainage[n] = free_drainage_
        # print('total free drainage at t =', t, 'is', total_free_drainage[n])
        
        deep_percolation_ = q_(rooting_depth)
        total_deep_percolation[n] = deep_percolation_
        print('Deep percolation at t =', t, 'is', total_deep_percolation[n])
        
        upper_flux_ = q_(uptake_zone_start)
        total_upper_flux[n] = upper_flux_
        print('Flux at upper boundary of uptake zone at t =', t, 'is', total_upper_flux[n])
        
        lower_flux_ = q_(uptake_zone_end)
        total_lower_flux[n] = lower_flux_
        print('Flux at lower boundary of uptake zone at t =', t, 'is', total_lower_flux[n])
        
        atmo_flux_ = q_(tp)
        total_atmo_flux[n] = atmo_flux_
        print('Flux at soil surface at t =', t, 'is', total_atmo_flux[n])
        
        S_ = vec_S(W, theta_vis_, theta1, theta2, theta3, theta4, Tp, NRLD, dx)
        total_uptake[n] = assemble(S_*dx)
        print('total uptake at t =', t, 'is', total_uptake[n])
        
        uptake_zone_water_content[n] = assemble(theta_vis_*chi_uptake_zone*dx)
        print('Uptake zone water content', uptake_zone_water_content[n])
    
    # Saving water loss and uptake quantities after full simulation.
    # np.savetxt(f'data/up_{name}_{ex_lab}T{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}_{imported_mesh}.txt', np.array(total_uptake))
    # np.savetxt(f'data/pr_{name}_{ex_lab}T{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}_{imported_mesh}.txt', np.array(total_precipitation))
    # np.savetxt(f'data/ev_{name}_{ex_lab}T{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}_{imported_mesh}.txt', np.array(total_evaporation))
    # np.savetxt(f'data/ro_{name}_{ex_lab}T{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}_{imported_mesh}.txt', np.array(total_runoff))
    # np.savetxt(f'data/fd_{name}_{ex_lab}T{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}_{imported_mesh}.txt', np.array(total_free_drainage))
    # np.savetxt(f'data/rzwc_{name}_{ex_lab}T{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}_{imported_mesh}.txt', np.array(root_zone_water_content))
    # np.savetxt(f'data/dp_{name}_{ex_lab}T{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}_{imported_mesh}.txt', np.array(total_deep_percolation))
    np.savetxt(f'data/uzwc_{name}_{ex_lab}T{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}_{imported_mesh}.txt', np.array(uptake_zone_water_content))
    np.savetxt(f'data/uf_{name}_{ex_lab}T{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}_{imported_mesh}.txt', np.array(total_upper_flux))
    # np.savetxt(f'data/af_{name}_{ex_lab}T{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}_{imported_mesh}.txt', np.array(total_atmo_flux))
    np.savetxt(f'data/lf_{name}_{ex_lab}T{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}_{imported_mesh}.txt', np.array(total_lower_flux))
    
    return total_uptake, total_evaporation, total_deep_percolation, total_free_drainage, total_precipitation, total_runoff, root_zone_water_content, uptake_zone_water_content, total_upper_flux, total_atmo_flux, total_lower_flux

patterns = np.array([1, 2, 3])
# patterns = np.array([1])
systems = ['trigo6days', 'trigo15days', 'trigo30days']
# systems = ['trigo30days']
precip_tots = np.array([0.12, 0.28])
# precip_tots = np.array([0.28])

for i in range(len(patterns)):
    pat = patterns[i]
    for j in range(len(systems)):
        system = systems[j]
        for k in range(len(precip_tots)):
            tot = precip_tots[k]
            tu_ex, te_ex, tdp_ex, tfd_ex, tpr_ex, tr_ex, rzwc_ex, uzwc_ex, tuf_ex, taf_ex, tlf_ex = simulator(system, 'sandy_loam', 'wetting', 0.069, True, 100, 3000, tot, pat, 0.5, 0.75, 3, 1E-10, 'eq', 'eq', 1.0)
            tu, te, tdp, tfd, tpr, tr, rzwc, uzwc, tuf, taf, tlf = simulator(system, 'sandy_loam', 'wetting', 0.069, False, 100, 3000, tot, pat, 0.5, 0.75, 3, 1E-10, 'eq', 'eq', 1.0)

# tu_ex, te_ex, tdp_ex, tpr_ex, tr_ex = simulator(system, 'sandy_loam', 'wetting', 0.069, True, 100, 3000, tot, pat, 0.5, 0.75, 3, 1E-10, 'eq', 'eq', 1.0)    
# tu, te, tdp, tpr, tr = simulator(system, 'sandy_loam', 'wetting', 0.069, False, 100, 3000, tot, pat, 0.5, 0.75, 3, 1E-10, 'eq', 'eq', 1.0)

