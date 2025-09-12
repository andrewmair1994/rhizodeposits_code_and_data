# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:44:22 2024

@author: andre
"""

# This is a code to plot the relationship between root age/depth, precipitation
# quantity, and percentage increase or decrease in cumulative uptake from
# effect of rhizodeposits.

###############################################################################
# Importing necessary libaries
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import math as math

###############################################################################
# Defining function to process data and output figure 7
###############################################################################

def post_processor(imported_mesh = 'def'):
    """
    

    Parameters
    ----------
    imported_mesh : String, optional
        3D mesh that is imported for the construction of root density functions.
        'def' imports the mesh that was constructed using the deprecated
        functionality mshr, 'box' imports the mesh that can still be
        created using available Legacy FEniCS docker images.

    Returns
    -------
    tot_up_diff : np.array(3, 6)
        Array showing the percentage change in total uptake that is induced by
        rhizodeposits for each precipitation regime and root system.
        Root system age decreases from row 1 to row 3 and quantity of 
        precipitation per event increases from column 1 to column 6.
    
    """
    
    # Ensuring mesh name entered is valid.
    if not (imported_mesh == 'def' or imported_mesh == 'box'):
        raise TypeError('Invalid entry for imported mesh')

    ###########################################################################
    # Importing all uptake data
    ###########################################################################

    # With rhizodeposits present total precipitation 0.12cm
    data_rhizodeposits6days_ppat1ptot0_12 = np.loadtxt(f'data/up_trigo6days_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat1ptot0_12Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_rhizodeposits6days_ppat2ptot0_12 = np.loadtxt(f'data/up_trigo6days_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat2ptot0_12Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_rhizodeposits6days_ppat3ptot0_12 = np.loadtxt(f'data/up_trigo6days_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat3ptot0_12Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_rhizodeposits15days_ppat1ptot0_12 = np.loadtxt(f'data/up_trigo15days_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat1ptot0_12Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_rhizodeposits15days_ppat2ptot0_12 = np.loadtxt(f'data/up_trigo15days_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat2ptot0_12Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_rhizodeposits15days_ppat3ptot0_12 = np.loadtxt(f'data/up_trigo15days_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat3ptot0_12Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_rhizodeposits30days_ppat1ptot0_12 = np.loadtxt(f'data/up_trigo30days_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat1ptot0_12Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_rhizodeposits30days_ppat2ptot0_12 = np.loadtxt(f'data/up_trigo30days_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat2ptot0_12Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_rhizodeposits30days_ppat3ptot0_12 = np.loadtxt(f'data/up_trigo30days_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat3ptot0_12Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')

    # With rhizodeposits present total precipitation 0.28cm
    data_rhizodeposits6days_ppat1ptot0_28 = np.loadtxt(f'data/up_trigo6days_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat1ptot0_28Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_rhizodeposits6days_ppat2ptot0_28 = np.loadtxt(f'data/up_trigo6days_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat2ptot0_28Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_rhizodeposits6days_ppat3ptot0_28 = np.loadtxt(f'data/up_trigo6days_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat3ptot0_28Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_rhizodeposits15days_ppat1ptot0_28 = np.loadtxt(f'data/up_trigo15days_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat1ptot0_28Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_rhizodeposits15days_ppat2ptot0_28 = np.loadtxt(f'data/up_trigo15days_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat2ptot0_28Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_rhizodeposits15days_ppat3ptot0_28 = np.loadtxt(f'data/up_trigo15days_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat3ptot0_28Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_rhizodeposits30days_ppat1ptot0_28 = np.loadtxt(f'data/up_trigo30days_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat1ptot0_28Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_rhizodeposits30days_ppat2ptot0_28 = np.loadtxt(f'data/up_trigo30days_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat2ptot0_28Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_rhizodeposits30days_ppat3ptot0_28 = np.loadtxt(f'data/up_trigo30days_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat3ptot0_28Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')

    # Without rhizodeposits present total precipitation 0.12cm
    data_no_rhizodeposits6days_ppat1ptot0_12 = np.loadtxt(f'data/up_trigo6days_no_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat1ptot0_12Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_no_rhizodeposits6days_ppat2ptot0_12 = np.loadtxt(f'data/up_trigo6days_no_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat2ptot0_12Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_no_rhizodeposits6days_ppat3ptot0_12 = np.loadtxt(f'data/up_trigo6days_no_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat3ptot0_12Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_no_rhizodeposits15days_ppat1ptot0_12 = np.loadtxt(f'data/up_trigo15days_no_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat1ptot0_12Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_no_rhizodeposits15days_ppat2ptot0_12 = np.loadtxt(f'data/up_trigo15days_no_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat2ptot0_12Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_no_rhizodeposits15days_ppat3ptot0_12 = np.loadtxt(f'data/up_trigo15days_no_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat3ptot0_12Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_no_rhizodeposits30days_ppat1ptot0_12 = np.loadtxt(f'data/up_trigo30days_no_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat1ptot0_12Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_no_rhizodeposits30days_ppat2ptot0_12 = np.loadtxt(f'data/up_trigo30days_no_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat2ptot0_12Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_no_rhizodeposits30days_ppat3ptot0_12 = np.loadtxt(f'data/up_trigo30days_no_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat3ptot0_12Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')

    # Without rhizodeposits present total precipitation 0.28cm
    data_no_rhizodeposits6days_ppat1ptot0_28 = np.loadtxt(f'data/up_trigo6days_no_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat1ptot0_28Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_no_rhizodeposits6days_ppat2ptot0_28 = np.loadtxt(f'data/up_trigo6days_no_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat2ptot0_28Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_no_rhizodeposits6days_ppat3ptot0_28 = np.loadtxt(f'data/up_trigo6days_no_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat3ptot0_28Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_no_rhizodeposits15days_ppat1ptot0_28 = np.loadtxt(f'data/up_trigo15days_no_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat1ptot0_28Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_no_rhizodeposits15days_ppat2ptot0_28 = np.loadtxt(f'data/up_trigo15days_no_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat2ptot0_28Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_no_rhizodeposits15days_ppat3ptot0_28 = np.loadtxt(f'data/up_trigo15days_no_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat3ptot0_28Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_no_rhizodeposits30days_ppat1ptot0_28 = np.loadtxt(f'data/up_trigo30days_no_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat1ptot0_28Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_no_rhizodeposits30days_ppat2ptot0_28 = np.loadtxt(f'data/up_trigo30days_no_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat2ptot0_28Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')
    data_no_rhizodeposits30days_ppat3ptot0_28 = np.loadtxt(f'data/up_trigo30days_no_exT3_0theta0_0_07cw0_eq_cd0_eq_sal_w0ppat3ptot0_28Nx100Nt3000tol10extot1_0_{imported_mesh}.txt')

    ###############################################################################
    # Calculating total uptakes
    ###############################################################################

    # With rhizodeposits present total precipitation 0.12cm
    tot_up_rhizodeposits6days_ppat1ptot0_12 = np.sum(data_rhizodeposits6days_ppat1ptot0_12)
    tot_up_rhizodeposits6days_ppat2ptot0_12 = np.sum(data_rhizodeposits6days_ppat2ptot0_12)
    tot_up_rhizodeposits6days_ppat3ptot0_12 = np.sum(data_rhizodeposits6days_ppat3ptot0_12)
    tot_up_rhizodeposits15days_ppat1ptot0_12 = np.sum(data_rhizodeposits15days_ppat1ptot0_12)
    tot_up_rhizodeposits15days_ppat2ptot0_12 = np.sum(data_rhizodeposits15days_ppat2ptot0_12)
    tot_up_rhizodeposits15days_ppat3ptot0_12 = np.sum(data_rhizodeposits15days_ppat3ptot0_12)
    tot_up_rhizodeposits30days_ppat1ptot0_12 = np.sum(data_rhizodeposits30days_ppat1ptot0_12)
    tot_up_rhizodeposits30days_ppat2ptot0_12 = np.sum(data_rhizodeposits30days_ppat2ptot0_12)
    tot_up_rhizodeposits30days_ppat3ptot0_12 = np.sum(data_rhizodeposits30days_ppat3ptot0_12)

    # With rhizodeposits present total precipitation 0.28cm
    tot_up_rhizodeposits6days_ppat1ptot0_28 = np.sum(data_rhizodeposits6days_ppat1ptot0_28)
    tot_up_rhizodeposits6days_ppat2ptot0_28 = np.sum(data_rhizodeposits6days_ppat2ptot0_28)
    tot_up_rhizodeposits6days_ppat3ptot0_28 = np.sum(data_rhizodeposits6days_ppat3ptot0_28)
    tot_up_rhizodeposits15days_ppat1ptot0_28 = np.sum(data_rhizodeposits15days_ppat1ptot0_28)
    tot_up_rhizodeposits15days_ppat2ptot0_28 = np.sum(data_rhizodeposits15days_ppat2ptot0_28)
    tot_up_rhizodeposits15days_ppat3ptot0_28 = np.sum(data_rhizodeposits15days_ppat3ptot0_28)
    tot_up_rhizodeposits30days_ppat1ptot0_28 = np.sum(data_rhizodeposits30days_ppat1ptot0_28)
    tot_up_rhizodeposits30days_ppat2ptot0_28 = np.sum(data_rhizodeposits30days_ppat2ptot0_28)
    tot_up_rhizodeposits30days_ppat3ptot0_28 = np.sum(data_rhizodeposits30days_ppat3ptot0_28)

    # Without rhizodeposits present total precipitation 0.12cm
    tot_up_no_rhizodeposits6days_ppat1ptot0_12 = np.sum(data_no_rhizodeposits6days_ppat1ptot0_12)
    tot_up_no_rhizodeposits6days_ppat2ptot0_12 = np.sum(data_no_rhizodeposits6days_ppat2ptot0_12)
    tot_up_no_rhizodeposits6days_ppat3ptot0_12 = np.sum(data_no_rhizodeposits6days_ppat3ptot0_12)
    tot_up_no_rhizodeposits15days_ppat1ptot0_12 = np.sum(data_no_rhizodeposits15days_ppat1ptot0_12)
    tot_up_no_rhizodeposits15days_ppat2ptot0_12 = np.sum(data_no_rhizodeposits15days_ppat2ptot0_12)
    tot_up_no_rhizodeposits15days_ppat3ptot0_12 = np.sum(data_no_rhizodeposits15days_ppat3ptot0_12)
    tot_up_no_rhizodeposits30days_ppat1ptot0_12 = np.sum(data_no_rhizodeposits30days_ppat1ptot0_12)
    tot_up_no_rhizodeposits30days_ppat2ptot0_12 = np.sum(data_no_rhizodeposits30days_ppat2ptot0_12)
    tot_up_no_rhizodeposits30days_ppat3ptot0_12 = np.sum(data_no_rhizodeposits30days_ppat3ptot0_12)

    # Without rhizodeposits present total precipitation 0.28cm
    tot_up_no_rhizodeposits6days_ppat1ptot0_28 = np.sum(data_no_rhizodeposits6days_ppat1ptot0_28)
    tot_up_no_rhizodeposits6days_ppat2ptot0_28 = np.sum(data_no_rhizodeposits6days_ppat2ptot0_28)
    tot_up_no_rhizodeposits6days_ppat3ptot0_28 = np.sum(data_no_rhizodeposits6days_ppat3ptot0_28)
    tot_up_no_rhizodeposits15days_ppat1ptot0_28 = np.sum(data_no_rhizodeposits15days_ppat1ptot0_28)
    tot_up_no_rhizodeposits15days_ppat2ptot0_28 = np.sum(data_no_rhizodeposits15days_ppat2ptot0_28)
    tot_up_no_rhizodeposits15days_ppat3ptot0_28 = np.sum(data_no_rhizodeposits15days_ppat3ptot0_28)
    tot_up_no_rhizodeposits30days_ppat1ptot0_28 = np.sum(data_no_rhizodeposits30days_ppat1ptot0_28)
    tot_up_no_rhizodeposits30days_ppat2ptot0_28 = np.sum(data_no_rhizodeposits30days_ppat2ptot0_28)
    tot_up_no_rhizodeposits30days_ppat3ptot0_28 = np.sum(data_no_rhizodeposits30days_ppat3ptot0_28)

    ###############################################################################
    # Creating array of rhizodeposit induced percentage change in total
    # uptake.
    ###############################################################################

    # Total uptakes with rhizodeposits
    tot_up_rhiz = np.array([[tot_up_rhizodeposits30days_ppat3ptot0_12, tot_up_rhizodeposits30days_ppat2ptot0_12, tot_up_rhizodeposits30days_ppat1ptot0_12, tot_up_rhizodeposits30days_ppat3ptot0_28, tot_up_rhizodeposits30days_ppat2ptot0_28, tot_up_rhizodeposits30days_ppat1ptot0_28],
                            [tot_up_rhizodeposits15days_ppat3ptot0_12, tot_up_rhizodeposits15days_ppat2ptot0_12, tot_up_rhizodeposits15days_ppat1ptot0_12, tot_up_rhizodeposits15days_ppat3ptot0_28, tot_up_rhizodeposits15days_ppat2ptot0_28, tot_up_rhizodeposits15days_ppat1ptot0_28],
                            [tot_up_rhizodeposits6days_ppat3ptot0_12, tot_up_rhizodeposits6days_ppat2ptot0_12, tot_up_rhizodeposits6days_ppat1ptot0_12, tot_up_rhizodeposits6days_ppat3ptot0_28, tot_up_rhizodeposits6days_ppat2ptot0_28, tot_up_rhizodeposits6days_ppat1ptot0_28]])

    # Total uptakes without rhizodeposits
    tot_up_no_rhiz = np.array([[tot_up_no_rhizodeposits30days_ppat3ptot0_12, tot_up_no_rhizodeposits30days_ppat2ptot0_12, tot_up_no_rhizodeposits30days_ppat1ptot0_12, tot_up_no_rhizodeposits30days_ppat3ptot0_28, tot_up_no_rhizodeposits30days_ppat2ptot0_28, tot_up_no_rhizodeposits30days_ppat1ptot0_28],
                               [tot_up_no_rhizodeposits15days_ppat3ptot0_12, tot_up_no_rhizodeposits15days_ppat2ptot0_12, tot_up_no_rhizodeposits15days_ppat1ptot0_12, tot_up_no_rhizodeposits15days_ppat3ptot0_28, tot_up_no_rhizodeposits15days_ppat2ptot0_28, tot_up_no_rhizodeposits15days_ppat1ptot0_28],
                               [tot_up_no_rhizodeposits6days_ppat3ptot0_12, tot_up_no_rhizodeposits6days_ppat2ptot0_12, tot_up_no_rhizodeposits6days_ppat1ptot0_12, tot_up_no_rhizodeposits6days_ppat3ptot0_28, tot_up_no_rhizodeposits6days_ppat2ptot0_28, tot_up_no_rhizodeposits6days_ppat1ptot0_28]])

    tot_up_diff = tot_up_rhiz - tot_up_no_rhiz
    
    print('difference in total uptake =', tot_up_diff)
  
    percentage_change = (tot_up_diff/tot_up_no_rhiz)*100

    percentage_change_rounded = np.round(percentage_change, 2)

    print('percentage change in cumulative uptake =', percentage_change)
    
    ###############################################################################
    # Plotting and setting plotting features
    ###############################################################################
    
    # Setting font for labels.
    font_label = {'family': 'serif',
                  'color':  'black',
                  'weight': 'normal',
                  'size': 14,
                  }

    # Setting font for ticks.
    font_ticks = {'family': 'serif',
                  'color':  'black',
                  'weight': 'normal',
                  'size': 11,
                  }

    # Setting font for legend.
    font_legend = font_manager.FontProperties(family='serif',
                                              weight='normal',
                                              style='normal', size=16)
    
    regimes = ['0.04 (3)', '0.06 (2)', '0.12 (1)', '0.093 (3)', '0.14 (2)', '0.28 (1)']
    system_age = ['30', '15', '6']
    
    fig, ax = plt.subplots()
    # im = ax.imshow(percentage_change, cmap = 'RdYlGn')
    im = ax.imshow(percentage_change, cmap = 'PiYG')

    for i in range(3): 
        for j in range(6):
            if percentage_change_rounded[i][j] >0:
                plt.annotate('+' + str(percentage_change_rounded[i][j])+'%', xy=(j, i), 
                              ha="center", va="center", color='black', font = 'serif')
            else:
                plt.annotate(str(percentage_change_rounded[i][j])+'%', xy=(j, i), 
                              ha="center", va="center", color='black', font = 'serif')

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(regimes)), labels=regimes)
    ax.set_yticks(np.arange(len(system_age)), labels=system_age)
    ax.tick_params(axis = 'both', labelsize = 11)
    # Changing font of ticks.
    for tick in ax.get_xticklabels():
        tick.set_fontname("serif")
        for tick in ax.get_yticklabels():
            tick.set_fontname("serif")

    ax.set_xlabel('Quantity of precipitation per event ($cm$)', fontdict = font_label)
    ax.set_ylabel('Root system age (days)', fontdict = font_label)

    cbar = ax.figure.colorbar(im, ax=ax, location = 'top', ticks = [-8.53, 0, 20, 40, 60, 80, 100, 115.95])
    cbar.ax.set_xticklabels(['-8.54', 0, '+20', '+40', '+60', '+80', '+100', '+115.95'], fontdict = font_ticks)
    cbar.ax.set_xlabel('Rhizodeposit-induced % change in total uptake ($cm$)', fontdict = font_label)

    plt.savefig(f'figures/figure_7_{imported_mesh}.eps')
    plt.savefig(f'figures/figure_7_{imported_mesh}.png')
    
    return tot_up_diff

post_processor()