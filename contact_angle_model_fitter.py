# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:20:04 2024

@author: andre
"""

import numpy as np
import math as math
from math import sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from scipy.optimize import curve_fit

###############################################################################
# Defining contact angle and concentration data.
# Data from Zickenrott et al,. (2016).
###############################################################################

# Contact angle with sand (no rhizodeposit present).
omega_w0 = 0.0

# Contact angle with sand (various concentrations of rhizodeposit present)
# 0.05 mg g^{-1} = 0.00005 mg mg^{-1}
omega_w1 = (10.0/180.0)*pi

# 0.10 mg g^{-1} = 0.0001 mg mg^{-1}
omega_w2 = (17.65/180.0)*pi        

# 0.20 mg g^{-1} = 0.0002 mg mg^{-1}
omega_w3 = (32.94/180.0)*pi

# 0.40 mg g^{-1} = 0.0004 mg mg^{-1}
omega_w4 = (30.59/180.0)*pi        

# 0.80 mg g^{-1} = 0.0008 mg mg^{-1}
omega_w5 = (28.24/180.0)*pi       

# 1.60 mg g^{-1} = 0.0016 mg mg^{-1}
omega_w6 = (35.29/180.0)*pi

# Concentrations of dried maize root rhizodeposits
concs = np.array([0.0, 0.00005, 0.0001, 0.0002, 0.0004, 0.0008, 0.0016])

angles = np.array([omega_w0, omega_w1, omega_w2, omega_w3, omega_w4, omega_w5, omega_w6])

###############################################################################
# Defining function type to be fitted for contact angle against concentration.
###############################################################################

def rhiz_contact_angle(x, a, b):
    """
    

    Parameters
    ----------
    x : Float
        Concentration or volumetric density values.
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
# Using curve_fit function to parametrise 
# rhiz_contact_angle against rhizodeposit concentration values and return the 
# optimal value for the parameters a and b in rhiz_contact_angle. 
###############################################################################

# Implementing the curve fitting tool. 
popt, pcov = curve_fit(rhiz_contact_angle, concs, angles)

print('Fitted parameter values =', popt)
print('Condition number of parameter covariance matrix (want this to be small) =', np.linalg.cond(pcov))
print('Diagonal elements of covariance matrix =', np.diag(pcov))
print('Actual angles =', angles)
print('Tensions from model =', rhiz_contact_angle(concs, popt[0], popt[1]))

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
ax.set_xlabel('Dried rhizodeposit concentration \n $c_d$ ($mg mg^{-1}$)', fontdict = font_label)
ax.set_ylabel('Contact angle \n $\omega$ ($rad$)', fontdict = font_label)
ax.xaxis.set_label_position('bottom')
ax.yaxis.set_label_position('left')
ax.set_xlim(0.0, 1.6e-3)
ax.set_ylim(0.0, 0.65)
ax.ticklabel_format(axis = 'x', scilimits = (1E-2, 0))

cont_concs = np.linspace(0, concs[-1], int(np.floor(0.0016/0.0001)))
ax.plot(cont_concs, rhiz_contact_angle(cont_concs, popt[0], popt[1]), 'g-', label = r'$\omega(c_D)$')
# ax.plot(concs, angles, 'r*', label = 'Data')
plt.legend(prop = font_legend, loc = 'lower right')
plt.tight_layout()

# Saving out the fitted parameter values.
np.savetxt('data/ca_vs_rhizodeposits_parameter_values_zickenrott2016.txt', np.array([popt[0], popt[1]]))

print("Maximum possible contact angle with rhizodeposits =", rhiz_contact_angle(0.0016, popt[0], popt[1]))
print("Maximum possible contact angle with rhizodeposits =", rhiz_contact_angle(0.016, popt[0], popt[1]))
print("Contact angle with no rhizodeposits =", rhiz_contact_angle(0.0, popt[0], popt[1]))
print((35.29/180.0)*pi)

plt.savefig(f'figures/contact_angle_vs_rhizodeposits.eps')
plt.savefig(f'figures/contact_angle_vs_rhizodeposits.png')
