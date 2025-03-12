# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:52:24 2024

@author: andre
"""

# This is a code to fit a model for surface tension as a function of wheat 
# root rhizodeposit concentration.

###############################################################################
# Importing necessary libraries
###############################################################################

import numpy as np
import math as math
from math import sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from scipy.optimize import curve_fit

###############################################################################
# Defining surface tension and concentration data.
###############################################################################

# Surface tension of pure water (mN m^{-1}).
# (Naveed et al. 2019). 
gamma0 = 72.86

# Read et al. 2003
# Surface tension (mN m^{-1}) at 0.12 mg cm^{-3}
gamma1 = 70.94

# Surface tension (mN m^{-1}) at 0.19 mg cm^{-3}
gamma2 = 70.0

# Surface tension (mN m^{-1}) at 0.31 mg cm^{-3}
gamma3 = 67.50

# Surface tension (mN m^{-1}) at 0.38 mg cm^{-3}
gamma4 = 55.31

# Surface tension (mN m^{-1}) at 0.5 mg cm^{-3}
gamma5 = 50.93

# Surface tension (mN m^{-1}) at 0.77 mg cm^{-3}
gamma6 = 50.63

# Surface tension (mN m^{-1}) at 0.96 mg cm^{-3}
gamma7 = 49.69

# Surface tension (mN m^{-1}) at 1.14 mg cm^{-3}
gamma8 = 48.44

# Surface tension (mN m^{-1}) at 2.07 mg cm^{-3}
gamma9 = 47.5

# Surface tension (mN m^{-1}) at 2.42 mg cm^{-3}
gamma10 = 47.65

concs = np.array([0, 0.12, 0.19, 0.31, 0.38, 0.5, 0.77, 0.96, 1.14, 2.07, 2.42])
tensions = np.array([gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6, gamma7, gamma8, gamma9, gamma10])

###############################################################################
# Defining function type to be fitted for surface tension against 
# rhizodeposit concentration.
###############################################################################

def rhiz_surf_tension(x, a, b):
    """
    

    Parameters
    ----------
    x : Float
        rhizodeposit concentration.
    a : Float
        Multiplier of power law.
    b : Float
        Shift in power law.

    Returns
    -------
    gamma: Float
        Surface tension of rhizodeposit solution

    """
   
    gamma = 47.5 + (72.86 - 47.5)/(1 + np.exp(a*(x-b)))
    
    return gamma

###############################################################################
# Using curve_fit function to parametrise 
# rhiz_surf_tension against rhizodeposit concentration values and return the 
# optimal value for the parameters a and b. 
###############################################################################

# Implementing the curve fitting tool. 
popt, pcov = curve_fit(rhiz_surf_tension, concs, tensions)

print('Fitted parameter values =', popt)
print('Condition number of parameter covariance matrix (want this to be small) =', np.linalg.cond(pcov))
print('Diagonal elements of covariance matrix =', np.diag(pcov))
print('Actual tensions =', tensions)
print('Tensions from model =', rhiz_surf_tension(concs, popt[0], popt[1]))

# Setting font for labels.
font_label = {'family': 'serif',
              'color':  'black',
              'weight': 'normal',
              'size': 20,
              }

# Setting font for legend.
font_legend = font_manager.FontProperties(family='serif',
                                          weight='normal',
                                          style='normal', size=20)

# Plotting the fitted curve for verification.
fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(8,4))

# Formatting
ax.tick_params(axis = 'both', labelsize = 18)
ax.set_xlabel('rhizodeposit concentration\nin solution $c_W$ (mg cm$^{-3}$)', fontdict = font_label)
ax.set_ylabel('Surface tension $\gamma$\n(mNm$^{-1}$)', fontdict = font_label)
ax.xaxis.set_label_position('bottom')
ax.yaxis.set_label_position('left')

cont_concs = np.linspace(0, concs[-1], int(np.floor(2.42/0.01)))
ax.plot(cont_concs, rhiz_surf_tension(cont_concs, popt[0], popt[1]), 'b-', label = r'$\gamma(c_W)$')
# ax.plot(concs, tensions, 'r*', label = 'Data')

ax.set_xlim(0, 2.42)
ax.set_ylim(40, 80)

# Changing font of ticks.
for tick in ax.get_xticklabels():
    tick.set_fontname("serif")
for tick in ax.get_yticklabels():
    tick.set_fontname("serif")
    
plt.legend(prop = font_legend, bbox_to_anchor = (1, 1))
plt.legend(prop = font_legend)
plt.tight_layout()

print("Surface tension at 0.5 mg cm^{-3} is \gamma =", rhiz_surf_tension(0.5, popt[0], popt[1]))
print("Surface tension with no rhizodeposits =", rhiz_surf_tension(0.0, popt[0], popt[1]))
print("Minimum surface tension with rhizodeposits =", rhiz_surf_tension(2.42, popt[0], popt[1]))

# Saving out the fitted parameter values.
np.savetxt('data/st_parameter_values_read_2003.txt', np.array([popt[0], popt[1]]))

plt.savefig('figures/surface_tension_vs_rhizodeposits.eps')
plt.savefig('figures/surface_tension_vs_rhizodeposits.png')

