# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:05:18 2023

@author: andre
"""

import numpy as np
import math as math
from math import sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from scipy.optimize import curve_fit

###############################################################################
# Entering viscosity and rhizodeposit concentration data
###############################################################################

# Viscosity of pure water 
# (Hosseini & Mossaddeghi 2024). 
mu0 = 0.00089

# Viscosity of water and wheat rhizodeposit solution (Pa.s or kg.m^{-1}.s^{-1}) 
# at 0.242mg cm^{-3} mucilage concentration (Hosseini & Mossaddeghi 2024).
mu1 = 0.00093

# Viscosity of water and wheat rhizodeposit solution (Pa.s or kg.m^{-1}.s^{-1}) 
# at 0.475 mg cm^{-3} mucilage concentration (Hosseini & Mossaddeghi 2024).
mu2 = 0.00096

# Viscosity of water and wheat rhizodeposit solution (Pa.s or kg.m^{-1}.s^{-1}) at 
# 1.983 mg cm^{-3} (Naveed et al. 2019).
mu3 = 0.00098

# Vectors of concentrations and assumed associated densities.
concs = np.array([0, 0.242, 0.475, 1.9832])

# Vector of viscosities.
viscs = np.array([mu0, mu1, mu2, mu3])

###############################################################################
# Defining a function for the evolution of viscosity against volumetric root
# density acting as a proxy for mucilage concentration.
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
    mu = mu0 + (mu3-mu0)*(1 + np.exp(-a*(x-b)))**(-10)
    
    return mu

###############################################################################
# Using curve_fit function to define a function that parametrises 
# rhiz_surf_tension against volumetric density values and returns the optimal
# parameter values.
###############################################################################

def viscosity_model_fitter(concentrations, viscosities):
    
    # Implementing the curve fitting tool. 
    popt, pcov = curve_fit(rhiz_viscosity, concentrations, viscosities)
    
    print('Fitted parameter values =', popt)
    print('Condition number of parameter covariance matrix (want this to be small) =', np.linalg.cond(pcov))
    print('Diagonal elements of covariance matrix =', np.diag(pcov))
    print('Actual viscosities =', viscosities)
    print('Viscosities from model =', rhiz_viscosity(concentrations, popt[0], popt[1]))
    
    # Setting font for labels.
    font_label = {'family': 'serif',
                  'color':  'black',
                  'weight': 'normal',
                  'size': 20,
                  }
    
    # Setting font for legend.
    font_legend = font_manager.FontProperties(family='serif',
                                              weight='normal',
                                              style='normal', size=16)
    
    # Plotting the fiited curve for verification.
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(8,4))
    
    # Formatting
    ax.tick_params(axis = 'both', labelsize = 20)
    ax.set_xlabel(r'Rhizodeposit concentration $c_w$ (mg cm$^{-3}$)', fontdict = font_label)
    ax.set_ylabel(r'Viscosity: $\mu$ (Pa$\cdot$s)', fontdict = font_label)
    ax.xaxis.set_label_position('bottom')
    ax.yaxis.set_label_position('left')
     
    cont_concs = np.linspace(concentrations[0], concentrations[-1], int(np.floor(concentrations[-1]/0.01)))
    ax.plot(cont_concs, rhiz_viscosity(cont_concs, popt[0], popt[1]), 'g-', label = r'$\mu(c_w)$')
    ax.plot(concentrations, rhiz_viscosity(concentrations, popt[0], popt[1]), 'b*')
    ax.plot(concentrations, viscosities, 'r*', label = 'Data')
    
    # ax.set_xlim(0, 10)
    # ax.set_ylim(0, 3)
    
    # Changing font of ticks.
    for tick in ax.get_xticklabels():
        tick.set_fontname("serif")
    for tick in ax.get_yticklabels():
        tick.set_fontname("serif")
        
    # plt.legend(prop = font_legend, bbox_to_anchor = (1, 1))
    plt.legend(prop = font_legend)
    plt.tight_layout()
    
    # Saving out the fitted parameter values.
    np.savetxt('data/mu_parameter_values_hosseini_mossaddeghi_2024.txt', np.array([popt[0], popt[1]]))

    plt.savefig('figures/viscosity_vs_rhizodeposits.eps')
    plt.savefig('figures/viscosity_vs_rhizodeposits.png')

            
    return popt
 
# Running function for fitting to viscosity data.
optimal_parameter_values = viscosity_model_fitter(concs, viscs) 


   