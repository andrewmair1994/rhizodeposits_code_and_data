# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 10:52:54 2025

@author: andre
"""

# This is a code that shows how rhizodeposit concentration affects the curves
# for water retention and soil hydraulic conductivity in our model setup.

###############################################################################
# Importing necessary libaries
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import math as math
from math import cos, pi

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
# Defining functions for the component parameters of the water retention
# and hydraulic conductivity curves. 
###############################################################################

# Function for the alpha shape parameter at a given point depending on 
# whether we are in a wetting or drying regime, and what level of rhizodeposit
# concentration there is.
def alpha_vg (status, alpha_vg_d, alpha_vg_w, a_st, b_st, a_ca, b_ca, cw, cd):
    """
    

    Parameters
    ----------
    status : Float
        Wetting or drying status at the current time.
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
        
    # If the soil is on a wetting trajectory.
    if status == alpha_vg_w:    
            
        # The wetting alpha is assigned.
        alpha_vg = alpha_vg_w*(gamma/gamma0)*(cos(omega_w0)/cos(omega_w))
        
    # The soil must be on a drying trajectory.
    else:
        # The drying alpha is assigned.
        alpha_vg = alpha_vg_d*(gamma0/gamma)*(cos(omega_w)/cos(omega_w0))
        
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

# Defining a function for the nodal saturated hydraulic conductivity.
def Ks(status, alpha_vg_d, Ksd, Ksw, a_st, b_st, a_ca, b_ca, a_vis, b_vis, cw, cd, beta):
    """
    

    Parameters
    ----------
    status : Float
        Indication of whether the soil is wetting or drying.
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
    
    # When we are in a drying curve.
    if status == alpha_vg_d:
        Ks = Ksd*pow(gamma0/gamma, beta)*(cos(omega_w)/cos(omega_w0))*(mu0/mu)
        
    # When we are in a wetting curve.
    else:
        Ks = Ksw*pow(gamma0/gamma, beta)*(cos(omega_w)/cos(omega_w0))*(mu0/mu)
    
    return Ks

# Regularisation constant for hydraulic conductivity function.
eps = 1E-15

# Tortuosity of porous medium.
l = 0.5

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

# Defining a function that returns the hydraulic conductivity.
def K_hyst(h, Ks, alpha_vg, n_vg, m_vg, theta_r0, theta_s0):
    """
    

    Parameters
    ----------
    h : Dolfin function
        Pressure head.
    Ks : Dolfin function
        Saturated hydraulic conductivity.
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
    
    return Ks*Kr(h, alpha_vg, n_vg, m_vg, theta_r0, theta_s0)

def plotter (func):
    """
    

    Parameters
    ----------
    func : String
        Indication of whether the water content or hydraulic conductivity is
        to be plotted. "theta" or "K

    """
    
    done = 'listo'
    
    # Ensuring function name entered is valid.
    if not (func == 'theta' or func == 'K'):
        raise TypeError('Invalid entry for "func". Should be "theta" or "K"')
        
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
    
    ###########################################################################
    # Setting values for soil hydraulic parameters in the absence of 
    # rhizodeposits.
    ###########################################################################
    
    # Residual water content.
    # (Carsel & Parish, 1988)
    theta_r0 = 0.065
    
    # Saturated water content.
    # (Carsel & Parish, 1988)
    theta_s0 = 0.41
    
    # Wetting inverse air entry pressure head cm^{-1} 
    # (Kool and Parker, 1987)
    alpha_vg_w = 0.0521
    print('Wetting alpha without rhizodeposits', alpha_vg_w)
    
    # Drying inverse air entry pressure head cm^{-1}  
    # (Kool and Parker, 1987)
    alpha_vg_d = 0.0114
    print('Drying alpha without rhizodeposits', alpha_vg_d)
    
    # Sandy loam (pore size parameter) 
    # (Carsel & Parish, 1988)
    n_vg = 1.89
    m_vg = 1-1/n_vg
    
    # Drying saturated hydraulic conductivity (cm/day) 
    # (Vogel & Zhang, 1996)
    Ksd = 5.0*24.0
    print('Drying Ks without rhizodeposits', Ksd)
    
    # Wetting saturated hydraulic conductivity (cm/day).
    # (Vogel & Zhang, 1996).
    Ksw = 3.0*24.0
    print('Wetting Ks without rhizodeposits', Ksw)
    
    # Defining maximum possible value of regularised relative conductivity.
    Kr_max = pow(1.0 - (eps/2.0)/(theta_s0 - theta_r0), l)*pow(1.0 - pow(1.0 - pow(1.0-(eps/2.0)/(theta_s0 - theta_r0), 1.0/m_vg), m_vg), 2)
    
    ###########################################################################
    # Reading in calibrated parameters for effects of rhizodeposits on soil 
    # hydraulic parameters.
    ###########################################################################
    
    beta = np.loadtxt(f'data/calibrated_parameter_values/beta.txt')
    # print('calibrated beta value = ', beta)
    
    ###########################################################################
    # Setting values for concentrations of adsorbed rhizodpeosits and 
    # rhizodeposits in solution.
    ###########################################################################
    
    # Concentration of rhizodeposits in solution mg cm^{-3}
    cw = 2.2
    
    # Concentration of adsorbed rhizodepositis mg mg^{-1}
    cd = 5.1e-4
    
    ###########################################################################
    # Setting range of pressure heads for observation
    ###########################################################################
    
    # h = np.linspace(-3596, 0, 3597)
    h = np.linspace(-4000, 0, 10)
    
    ###########################################################################
    # Setting parameter values for wetting and drying curves with
    # rhizodeposit effect incorporated.
    ###########################################################################
    
    # Wetting alpha
    alpha_vg_w_rhiz = alpha_vg(alpha_vg_w, alpha_vg_d, alpha_vg_w, a_st, b_st, a_ca, b_ca, cw, cd) 
    print('Wetting alpha with rhizodeposits', alpha_vg_w_rhiz)
        
    # Drying alpha 
    alpha_vg_d_rhiz = alpha_vg(alpha_vg_d, alpha_vg_d, alpha_vg_w, a_st, b_st, a_ca, b_ca, cw, cd)
    print('Drying alpha with rhizodeposits', alpha_vg_d_rhiz)
    
    # Wetting Ks
    Ksw_rhiz = Ks(alpha_vg_w, alpha_vg_d, Ksd, Ksw, a_st, b_st, a_ca, b_ca, a_vis, b_vis, cw, cd, beta)
    print('Wetting Ks with rhizodeposits', Ksw_rhiz)
    
    # Drying Ks
    Ksd_rhiz = Ks(alpha_vg_d, alpha_vg_d, Ksd, Ksw, a_st, b_st, a_ca, b_ca, a_vis, b_vis, cw, cd, beta)
    print('Drying Ks with rhizodeposits', Ksd_rhiz)
    
    ########################################################################### 
    # Obtaining values for plotting
    ###########################################################################
    
    ###########################################################################
    # Curves of retention function with and without rhizodeposit influence
    ###########################################################################
    
    theta_w_rhiz = theta_hyst(h, alpha_vg_w_rhiz, theta_r0, theta_s0, n_vg, m_vg)
    print('Wetting water retention curve with rhizodeposit influence', theta_w_rhiz)
    
    theta_w = theta_hyst(h, alpha_vg_w, theta_r0, theta_s0, n_vg, m_vg)
    print('Wetting water retention curve without rhizodeposit influence', theta_w)
    
    theta_d_rhiz = theta_hyst(h, alpha_vg_d_rhiz, theta_r0, theta_s0, n_vg, m_vg)
    print('Drying water retention curve with rhizodeposit influence', theta_d_rhiz)
    
    theta_d = theta_hyst(h, alpha_vg_d, theta_r0, theta_s0, n_vg, m_vg)
    print('Drying water retention curve without rhizodeposit influence', theta_d)
    
    ###########################################################################
    # Curves of hydraulic conductivity function with and without rhizodeposit 
    # influence
    ########################################################################### 
    
    K_w_rhiz = K_hyst(h, Ksw_rhiz, alpha_vg_w_rhiz, n_vg, m_vg, theta_r0, theta_s0)
    print('Wetting hydraulic conductivity curve with rhizodeposit influence', K_w_rhiz)
    
    K_w = K_hyst(h, Ksw, alpha_vg_w, n_vg, m_vg, theta_r0, theta_s0)
    print('Wetting hydraulic conductivity curve without rhizodeposit influence', K_w)
    
    K_d_rhiz = K_hyst(h, Ksd_rhiz, alpha_vg_d_rhiz, n_vg, m_vg, theta_r0, theta_s0)
    print('Drying hydraulic conductivity curve with rhizodeposit influence', K_d_rhiz)
    
    K_d = K_hyst(h, Ksd, alpha_vg_d, n_vg, m_vg, theta_r0, theta_s0)
    print('Wetting hydraulic conductivity curve without rhizodeposit influence', K_d)
    
    ###########################################################################
    # Plotting and saving
    ###########################################################################
    
    # Setting font for labels.
    font_label = {'family': 'serif',
                  'color':  'black',
                  'weight': 'normal',
                  'size': 15,
                  }

    # Setting font for legend.
    font_legend = font_manager.FontProperties(family='serif',
                                              weight='normal',
                                              style='normal', size=15)
    
    # Initiating figure.
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(8,4))
    
    # Formatting.
    ax.tick_params(axis = 'both', labelsize = 16)
    ax.set_xlabel('Pressure head $h$ (cm)', fontdict = font_label)
    
    ax.xaxis.set_label_position('bottom')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    
    if func == 'theta':
        ax.plot(h, theta_w_rhiz, 'r-', label = 'Wetting $c_{w} = c_{w,0}, c_{d} = c_{d,0}$')
        ax.plot(h, theta_w, 'b-', label = 'Wetting $c_{w} = c_{d} = 0$')
        ax.plot(h, theta_d_rhiz, 'r--', label = 'Drying $c_{w} = c_{w,0}, c_{d} = c_{d,0}$')
        ax.plot(h, theta_d, 'b--', label = 'Drying $c_{w} = c_{d} = 0$')
        ax.set_xlim(-4000, 0)
        ax.set_ylim(0.069, 0.41)
        ax.set_ylabel('Water content ($cm^{3}\ cm^{-3}$)', fontdict = font_label) 
        plt.legend(prop = font_legend)
        
    else:
        ax.plot(h, np.log(K_w_rhiz), 'r-', label = 'Wetting $c_{w} = c_{w,0}, c_{d} = c_{d,0}$')
        ax.plot(h, np.log(K_w), 'b-', label = 'Wetting $c_{w} = c_{d} = 0$')
        ax.plot(h, np.log(K_d_rhiz), 'r--', label = 'Drying $c_{w} = c_{w,0}, c_{d} = c_{d,0}$')
        ax.plot(h, np.log(K_d), 'b--', label = 'Drying $c_{w} = c_{d} = 0$')
        ax.set_xlim(-4000, 0)
        ax.set_ylim(-20, 10)
        ax.set_ylabel('log$(K(h))$ ($cm\ d^{-1}$)', fontdict = font_label)
        plt.legend(prop = font_legend)
        
    # Changing font of ticks.
    for tick in ax.get_xticklabels():
        tick.set_fontname("serif")
    for tick in ax.get_yticklabels():
        tick.set_fontname("serif")
         
    # plt.legend(prop = font_legend, bbox_to_anchor = (1, 1))
    plt.tight_layout()
    
    plt.savefig(f'figures/plot_rhizodeposit_effect_{func}.eps')
    plt.savefig(f'figures/plot_rhizodeposit_effect_{func}.png')
            
    return done

done = plotter('theta')
