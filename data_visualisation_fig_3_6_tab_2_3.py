# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:01:51 2024

@author: andre
"""

#  This is a code to plot Figures 3 to 6 in the manuscript and to generate the
# cumulative uptake values that are shown in Tables 2 and 3 of the manuscript.

###############################################################################
# Importing necessary libaries
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import math as math

###############################################################################
# Defining function to process data and output figures/table values.
###############################################################################


def post_processor(figure,
                   name,  
                   soil_type, 
                   status0, 
                   theta_init, 
                   Nx, 
                   Nt, 
                   p_tot, 
                   p_pat, 
                   T, 
                   tol, 
                   cw_init, 
                   cd_init,
                   ex_total,
                   ):
    """
    

    Parameters
    ----------
    figure : String
        The quantity that you want to plot: 
        evaporation, net_surface_flux, deep_percolation, mid-percolation, 
        runoff, total water loss, precipitation, uptake, total_water_content
    name : String
        Specific plant age or all ages at the same time. For all plant ages at
        the same time, enter 'all'.
    soil_type : String
        Type of soil.
    status0 : String
        'wetting' or 'drying'
    theta_init : Float:
        Initial water content within domain
    Nx : Integer
        Level of spatial refinement.
    Nt : Integer
        Number of timesteps.
    p_tot : FLoat
        Total rainfall amount.
    p_pat : Integer
        Number of rainfall events: 1, 2, 3, 5, 10.
    T : Float
        Final time.
    tol : Float expressed as power (e.g. 1E-1)
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
        dried rhizodeposit concentration.
    Returns
    -------
    summed_quantities : FLoat
        Cumulative sum of quantity of interest over time.

    """
    
    ###########################################################################
    # Checking that input values are correct
    ###########################################################################
    
    # Writing an error message if an incompatible entry has been given in the
    # figure input.
    if figure != 'evaporation' and figure != 'deep_percolation' and figure != 'runoff' and figure != 'total_water_loss' and figure != 'uptake' and figure != 'precipitation' and figure != 'total_water_content':
        raise TypeError('The only valid inputs for figure are: evaporation, deep_percolation, runoff, total_water_loss, uptake or precipitation')
    
    if figure == 'evaporation':
        fig_tag = 'ev'
    
    elif figure == 'deep_percolation':
        fig_tag = 'dp'
        
    elif figure == 'runoff':
        fig_tag = 'ro'
    
    elif figure == 'uptake':
        fig_tag = 'up'
        
    elif figure == 'precipitation':
        fig_tag = 'pr'
    
    elif figure == 'total_water_content':
        fig_tag = 'wc'    
    
    else:
        fig_tag = 'twl'
    
    # Insuring soil type entered is valid.
    if not (soil_type == 'sandy_loam' or soil_type == 'loamy_sand'):
        raise TypeError('Invalid soil type entered') 
    
    # Insuring initial status entered is valid.
    if not (status0 == 'wetting' or status0 == 'drying'):
        raise TypeError('Invalid initial wetting/drying status entered')
    
    # Insuring precipitation pattern is valid.
    if not (p_pat == 3 or p_pat == 2 or p_pat == 1):
        raise TypeError('Invalid precipitation pattern entered')
        
    ###########################################################################
    # Entering soil hydraulic parameters so that ex_total label can be 
    # correctly calculated in the case that tthe initial saturated and dried
    # rhizodeposit concentrations are explicitly prescribed.
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
    
    # Creating labels for level of spatial and temporal discretisation. 
    Nx_lab = str(math.modf(Nx)[1])[:-2]
    Nt_lab = str(math.modf(Nt)[1])[:-2]
    
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
        
    # Writing a label for the final time of the simulation
    T_int = str(math.modf(T)[1])[:-2]
    T_dec = str(np.round(math.modf(T)[0], 2))[2:]

    # Writing a label for the total precipitation
    p_tot_int = str(math.modf(p_tot)[1])[:-2]
    p_tot_dec = str(np.round(math.modf(p_tot)[0], 2))[2:]
    
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
    
    if not (cw_init == 'eq' and cd_init == 'eq'):
        ex_total = theta_init*cw_init + rho*cd_init
        
    ex_total_int = str(math.modf(ex_total)[1])[:-2]
    ex_total_dec = str(np.round(math.modf(ex_total)[0], 2))[2:]
        
    # Writing labels to indicate if initial state of soil was wetting or drying
    if status0 == 'wetting':
        stat_lab = 'w0'
        
    else:
        stat_lab = 'd0'
        
    # Computing timestep size.
    tau = T/Nt
    
    if ((figure == 'evaporation'
        or figure == 'deep_percolation' 
        or figure == 'runoff'
        or figure == 'uptake'
        or figure == 'precipitation'
        or figure == 'total_water_content')
        and name != 'all'):
    
        data_rhizodeposits = np.loadtxt(f'data/{fig_tag}_{name}_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')
        data_no_rhizodeposits = np.loadtxt(f'data/{fig_tag}_{name}_no_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')
            
        # Computing cumulative totals.
        cum_data_rhizodeposits = np.cumsum(tau*data_rhizodeposits)
        cum_data_no_rhizodeposits = np.cumsum(tau*data_no_rhizodeposits)
        
        summed_quantity_rhizodeposits = cum_data_rhizodeposits[-1]
        summed_quantity_no_rhizodeposits = cum_data_no_rhizodeposits[-1]
    
    elif ((figure == 'evaporation'
           or figure == 'deep_percolation'  
           or figure == 'runoff'
           or figure == 'uptake'
           or figure == 'precipitation'
           or figure == 'total_water_content')
          and name == 'all'):
        
        data_rhizodeposits6days = np.loadtxt(f'data/{fig_tag}_trigo6days_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')
        data_rhizodeposits15days = np.loadtxt(f'data/{fig_tag}_trigo15days_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')
        data_rhizodeposits30days = np.loadtxt(f'data/{fig_tag}_trigo30days_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')
        data_no_rhizodeposits6days = np.loadtxt(f'data/{fig_tag}_trigo6days_no_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')
        data_no_rhizodeposits15days = np.loadtxt(f'data/{fig_tag}_trigo15days_no_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')
        data_no_rhizodeposits30days = np.loadtxt(f'data/{fig_tag}_trigo30days_no_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')
            
        # Computing cumulative totals.
        cum_data_rhizodeposits6days = np.cumsum(tau*data_rhizodeposits6days)
        cum_data_rhizodeposits15days = np.cumsum(tau*data_rhizodeposits15days)
        cum_data_rhizodeposits30days = np.cumsum(tau*data_rhizodeposits30days)
        cum_data_no_rhizodeposits6days = np.cumsum(tau*data_no_rhizodeposits6days)
        cum_data_no_rhizodeposits15days = np.cumsum(tau*data_no_rhizodeposits15days)
        cum_data_no_rhizodeposits30days = np.cumsum(tau*data_no_rhizodeposits30days)
        
        summed_quantity_rhizodeposits6days = cum_data_rhizodeposits6days[-1]
        summed_quantity_rhizodeposits15days = cum_data_rhizodeposits15days[-1]
        summed_quantity_rhizodeposits30days = cum_data_rhizodeposits30days[-1]
        summed_quantity_no_rhizodeposits6days = cum_data_no_rhizodeposits6days[-1]
        summed_quantity_no_rhizodeposits15days = cum_data_no_rhizodeposits15days[-1]
        summed_quantity_no_rhizodeposits30days = cum_data_no_rhizodeposits30days[-1] 
        
    elif(figure == 'total_water_loss'
         and name != 'all'):
        ev_rhizodeposits = np.loadtxt(f'data/ev_{name}_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')    
        ro_rhizodeposits = np.loadtxt(f'data/ro_{name}_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt') 
        dp_rhizodeposits = np.loadtxt(f'data/dp_{name}_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt') 
        ev_no_rhizodeposits = np.loadtxt(f'data/ev_{name}_no_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')    
        ro_no_rhizodeposits = np.loadtxt(f'data/ro_{name}_no_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')    
        dp_no_rhizodeposits = np.loadtxt(f'data/dp_{name}_no_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')
    
        twl_rhizodeposits = ev_rhizodeposits + dp_rhizodeposits + ro_rhizodeposits
        twl_no_rhizodeposits = ev_no_rhizodeposits + dp_no_rhizodeposits + ro_no_rhizodeposits  
        
        # Computing cumulative totals.
        cum_twl_rhizodeposits = np.cumsum(tau*twl_rhizodeposits)
        cum_twl_no_rhizodeposits = np.cumsum(tau*twl_no_rhizodeposits)
        # 
        summed_quantity_rhizodeposits = cum_twl_rhizodeposits[-1]
        summed_quantity_no_rhizodeposits = cum_twl_no_rhizodeposits[-1]
        
    else:
        ev_rhizodeposits6days = np.loadtxt(f'data/ev_trigo6days_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')    
        ev_rhizodeposits15days = np.loadtxt(f'data/ev_trigo15days_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')
        ev_rhizodeposits30days = np.loadtxt(f'data/ev_trigo30days_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')
        dp_rhizodeposits6days = -1*np.loadtxt(f'data/dp_trigo6days_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')    
        dp_rhizodeposits15days = -1*np.loadtxt(f'data/dp_trigo15days_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')
        dp_rhizodeposits30days = -1*np.loadtxt(f'data/dp_trigo30days_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')
        ro_rhizodeposits6days = np.loadtxt(f'data/ro_trigo6days_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')    
        ro_rhizodeposits15days = np.loadtxt(f'data/ro_trigo15days_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt') 
        ro_rhizodeposits30days = np.loadtxt(f'data/ro_trigo30days_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')
        ev_no_rhizodeposits6days = np.loadtxt(f'data/ev_trigo6days_no_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')    
        ev_no_rhizodeposits15days = np.loadtxt(f'data/ev_trigo15days_no_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')
        ev_no_rhizodeposits30days = np.loadtxt(f'data/ev_trigo30days_no_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')
        dp_no_rhizodeposits6days = -1*np.loadtxt(f'data/dp_trigo6days_no_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')
        dp_no_rhizodeposits15days = -1*np.loadtxt(f'data/dp_trigo15days_no_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')
        dp_no_rhizodeposits30days = -1*np.loadtxt(f'data/dp_trigo30days_no_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')
        ro_no_rhizodeposits6days = np.loadtxt(f'data/ro_trigo6days_no_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt')
        ro_no_rhizodeposits15days = np.loadtxt(f'data/ro_trigo15days_no_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt') 
        ro_no_rhizodeposits30days = np.loadtxt(f'data/ro_trigo30days_no_exT{T_int}_{T_dec}theta0_{theta_init_int}_{theta_init_dec}cw0_{cw_init_int}_{cw_init_dec}cd0_{cd_init_int}_{cd_init_dec}{st_tag}_{stat_lab}ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}Nx{Nx_lab}Nt{Nt_lab}tol{tol_tag}extot{ex_total_int}_{ex_total_dec}.txt') 
        
        twl_rhizodeposits6days = ev_rhizodeposits6days + dp_rhizodeposits6days + ro_rhizodeposits6days
        twl_rhizodeposits15days = ev_rhizodeposits15days + dp_rhizodeposits15days + ro_rhizodeposits15days
        twl_rhizodeposits30days = ev_rhizodeposits30days + dp_rhizodeposits30days + ro_rhizodeposits30days
        twl_no_rhizodeposits6days = ev_no_rhizodeposits6days + dp_no_rhizodeposits6days + ro_no_rhizodeposits6days
        twl_no_rhizodeposits15days = ev_no_rhizodeposits15days + dp_no_rhizodeposits15days + ro_no_rhizodeposits15days
        twl_no_rhizodeposits30days = ev_no_rhizodeposits30days + dp_no_rhizodeposits30days + ro_no_rhizodeposits30days
        
        # Computing cumulative totals.
        cum_twl_rhizodeposits6days = np.cumsum(tau*twl_rhizodeposits6days)
        cum_twl_rhizodeposits15days = np.cumsum(tau*twl_rhizodeposits15days)
        cum_twl_rhizodeposits30days = np.cumsum(tau*twl_rhizodeposits30days)
        cum_twl_no_rhizodeposits6days = np.cumsum(tau*twl_no_rhizodeposits6days)
        cum_twl_no_rhizodeposits15days = np.cumsum(tau*twl_no_rhizodeposits15days)
        cum_twl_no_rhizodeposits30days = np.cumsum(tau*twl_no_rhizodeposits30days)
        
        # 
        summed_quantity_rhizodeposits6days = cum_twl_rhizodeposits6days[-1]
        summed_quantity_rhizodeposits15days = cum_twl_rhizodeposits15days[-1]
        summed_quantity_rhizodeposits30days = cum_twl_rhizodeposits30days[-1]
        summed_quantity_no_rhizodeposits6days = cum_twl_no_rhizodeposits6days[-1]
        summed_quantity_no_rhizodeposits15days = cum_twl_no_rhizodeposits15days[-1]
        summed_quantity_no_rhizodeposits30days = cum_twl_no_rhizodeposits30days[-1]
        
    if name != 'all':
        summed_quantities = np.array([summed_quantity_rhizodeposits, summed_quantity_no_rhizodeposits])
        
    else:
        summed_quantities6days = np.array([summed_quantity_rhizodeposits6days, summed_quantity_no_rhizodeposits6days])
        summed_quantities15days = np.array([summed_quantity_rhizodeposits15days, summed_quantity_no_rhizodeposits15days])
        summed_quantities30days = np.array([summed_quantity_rhizodeposits30days, summed_quantity_no_rhizodeposits30days])
        
        summed_quantities = [summed_quantities6days, summed_quantities15days, summed_quantities30days]
        
    ###########################################################################
    # Constructing figures as chosen in function input. 
    ###########################################################################
    
    # Setting vector containing the timestep values of the simulation.
    time = np.linspace(0, T, T*1000)

    # Setting font for labels.
    font_label = {'family': 'serif',
                  'color':  'black',
                  'weight': 'normal',
                  'size': 17.5,
                  }

    # Setting font for legend.
    font_legend = font_manager.FontProperties(family='serif',
                                              weight='normal',
                                              style='normal', size=14)

    # Initiating figure
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(8,4))
    
    # Formatting
    ax.tick_params(axis = 'both', labelsize = 20)
    ax.set_xlabel('Time ($d$)', fontdict = font_label)

    ax.xaxis.set_label_position('bottom')
    ax.yaxis.set_label_position('left')
    
    if figure == 'evaporation' and name != 'all':
        ax.plot(time, data_rhizodeposits, 'r-', label = 'Rhizodeposits')
        # ax.plot(time, data_no_rhizodeposits, 'b--', label = 'No rhizodeposits')
        ax.set_xlim(0, 3)
        ax.set_ylim(0.0, 0.062)
        ax.set_ylabel('Evaporation rate ($cm\ d^{-1}$)', fontdict = font_label)        
        
    if figure == 'evaporation' and name == 'all':
        # ax.plot(time, data_rhizodeposits6days, 'r-', label = 'Rhiz 6-days old ')
        # ax.plot(time, data_rhizodeposits15days, 'r--', label = 'Rhiz 15-days old')
        # ax.plot(time, data_rhizodeposits30days, 'r:', label = 'Rhiz 30-days old')
        # ax.plot(time, data_no_rhizodeposits6days, 'b-', label = 'No rhiz 6-days old')
        # ax.plot(time, data_no_rhizodeposits15days, 'b--', label = 'No rhiz 15-days old')
        # ax.plot(time, data_no_rhizodeposits30days, 'b:', label = 'No rhiz 30-days old')
        ax.plot(time, data_rhizodeposits6days, 'r-')
        ax.plot(time, data_rhizodeposits15days, 'r--')
        ax.plot(time, data_rhizodeposits30days, 'r:')
        ax.plot(time, data_no_rhizodeposits6days, 'b-')
        ax.plot(time, data_no_rhizodeposits15days, 'b--')
        ax.plot(time, data_no_rhizodeposits30days, 'b:')
        ax.set_xlim(0, 3)
        ax.set_ylim(0.0, 0.062)
        ax.set_ylabel('Evaporation rate ($cm\ d^{-1}$)', fontdict = font_label)
        # plt.legend(prop = font_legend)
        
    elif figure == 'deep_percolation' and name != 'all':
        ax.plot(time, data_rhizodeposits, 'r-', label = 'Rhizodeposits')
        ax.plot(time, data_no_rhizodeposits, 'b--', label = 'No rhizodeposits')
        ax.set_ylabel('Deep percolation ($cmd^{-1}$)', fontdict = font_label)
    
    elif figure == 'deep_percolation' and name == 'all':
        ax.plot(time, data_rhizodeposits6days, 'r-', label = 'Rhizodeposits')
        ax.plot(time, data_rhizodeposits15days, 'r--')
        ax.plot(time, data_rhizodeposits30days, 'r:')
        ax.plot(time, data_no_rhizodeposits6days, 'b-', label = 'No rhizodeposits')
        ax.plot(time, data_no_rhizodeposits15days, 'b--')
        ax.plot(time, data_no_rhizodeposits30days, 'b:')
        ax.set_ylabel('Deep percolation ($cmd^{-1}$)', fontdict = font_label)
        
    elif figure == 'runoff' and name != 'all':
        ax.plot(time, data_rhizodeposits, 'r-', label = 'Rhizodeposits')
        ax.plot(time, data_no_rhizodeposits, 'b--', label = 'No rhizodeposits')
        ax.set_ylabel('Runoff ($cmd^{-1}$)', fontdict = font_label)
        
    elif figure == 'runoff' and name == 'all':
        ax.plot(time, data_rhizodeposits6days, 'r-', label = 'Rhizodeposits')
        ax.plot(time, data_rhizodeposits15days, 'r--')
        ax.plot(time, data_rhizodeposits30days, 'r:')
        ax.plot(time, data_no_rhizodeposits6days, 'b-', label = 'No rhizodeposits')
        ax.plot(time, data_no_rhizodeposits15days, 'b--')
        ax.plot(time, data_no_rhizodeposits30days, 'b:')
        ax.set_ylabel('Runoff ($cmd^{-1}$)', fontdict = font_label)
        
    elif figure == 'uptake' and name != 'all':
        ax.plot(time, data_rhizodeposits, 'r-', label = 'Rhizodeposits')
        ax.plot(time, data_no_rhizodeposits, 'b--', label = 'No rhizodeposits')
        ax.set_xlim(0, 3)
        ax.set_ylim(0.0, 1.25e-2)
        ax.set_ylabel('Uptake rate ($cm\ d^{-1}$)', fontdict = font_label)
        ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
    
    elif figure == 'uptake' and name == 'all':
        ax.plot(time, data_rhizodeposits6days, 'r-')
        ax.plot(time, data_rhizodeposits15days, 'r--')
        ax.plot(time, data_rhizodeposits30days, 'r:')
        ax.plot(time, data_no_rhizodeposits6days, 'b-')
        ax.plot(time, data_no_rhizodeposits15days, 'b--')
        ax.plot(time, data_no_rhizodeposits30days, 'b:')
        ax.set_xlim(0, 3)
        ax.set_ylim(0.0, 1.25e-2)
        ax.set_ylabel('Uptake rate ($cm\ d^{-1}$)', fontdict = font_label)
        ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
    
    elif figure == 'total_water_content' and name != 'all':
        ax.plot(time, data_rhizodeposits, 'r-', label = 'Rhizodeposits')
        ax.plot(time, data_no_rhizodeposits, 'b--', label = 'No rhizodeposits')
        ax.set_ylabel('Total water ($cm$)', fontdict = font_label)
    
    elif figure == 'total_water_content' and name == 'all':
        ax.plot(time, data_rhizodeposits6days, 'r-', label = 'Rhizodeposits')
        ax.plot(time, data_rhizodeposits15days, 'r--')
        ax.plot(time, data_rhizodeposits30days, 'r:')
        ax.plot(time, data_no_rhizodeposits6days, 'b-', label = 'No rhizodeposits')
        ax.plot(time, data_no_rhizodeposits15days, 'b--')
        ax.plot(time, data_no_rhizodeposits30days, 'b:')
    
    elif figure == 'precipitation' and name != 'all':
        ax.plot(time, data_rhizodeposits, 'g-')
        ax.plot(time, data_no_rhizodeposits, 'g-')
        ax.set_ylabel('Precipitation ($cmd^{-1}$)', fontdict = font_label)
        ax.set_xlim(0, 3)
        ax.set_ylim(0.0, 0.6)
    
    elif figure == 'precipitation' and name == 'all':
        # ax.plot(time, data_rhizodeposits6days, 'g-')
        ax.plot(time, data_no_rhizodeposits6days, 'g-')
        ax.set_ylabel('Precipitation rate ($cm\ d^{-1}$)', fontdict = font_label)    
        ax.set_xlim(0, 3)
        ax.set_ylim(0.0, 0.6)
        # ax.tick_params(left = False, labelleft = False)
        # ax.tick_params(bottom = False, labelbottom = False)
    
    elif figure == 'total_water_loss' and name != 'all':
        ax.plot(time, twl_rhizodeposits, 'r-', label = 'Rhizodeposits')
        ax.plot(time, twl_no_rhizodeposits, 'b--', label = 'No rhizodeposits')
        ax.set_ylabel('Total water loss (cmd$^{-1}$)', fontdict = font_label)    
    
    elif figure == 'total_water_loss' and name == 'all':
        ax.plot(time, twl_rhizodeposits6days, 'r-', label = 'Rhizodeposits')
        ax.plot(time, twl_rhizodeposits15days, 'r--')
        ax.plot(time, twl_rhizodeposits30days, 'r:')
        ax.plot(time, twl_no_rhizodeposits6days, 'b-', label = 'No rhizodeposits')
        ax.plot(time, twl_no_rhizodeposits15days, 'b--')
        ax.plot(time, twl_no_rhizodeposits30days, 'b:')
    
    # Changing font of ticks.
    for tick in ax.get_xticklabels():
        tick.set_fontname("serif")
    for tick in ax.get_yticklabels():
        tick.set_fontname("serif")
         
    # plt.legend(prop = font_legend, bbox_to_anchor = (1, 1))
    plt.tight_layout()
    
    plt.savefig(f'figures/{fig_tag}_ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}.eps')
    plt.savefig(f'figures/{fig_tag}_ppat{p_tag}ptot{p_tot_int}_{p_tot_dec}.png')
    
    if name != 'all' and figure != 'total_water_loss':
        print('cumulative', figure, 'rhizodeposits =', summed_quantities[0])
        print('cumulative', figure, 'no rhizodeposits =', summed_quantities[1])
    
        print('Max rate', figure, 'rhizodeposits =', np.amax(data_rhizodeposits))
        print('Min rate', figure, 'rhizodeposits =', np.amin(data_rhizodeposits))
        print('Max rate', figure, 'no rhizodeposits =', np.amax(data_no_rhizodeposits))
        print('Min rate', figure, 'no rhizodeposits =', np.amin(data_no_rhizodeposits))
    
    elif name != 'all' and figure == 'total_water_loss':
        print('cumulative', figure, 'rhizodeposits =', summed_quantities[0])
        print('cumulative', figure, 'no rhizodeposits =', summed_quantities[1])
    
        print('Max rate', figure, 'rhizodeposits =', np.amax(twl_rhizodeposits))
        print('Min rate', figure, 'rhizodeposits =', np.amin(twl_rhizodeposits))
        print('Max rate', figure, 'no rhizodeposits =', np.amax(twl_no_rhizodeposits))
        print('Min rate', figure, 'no rhizodeposits =', np.amin(twl_no_rhizodeposits))
        
    elif name == 'all' and figure != 'total_water_loss':
        print('cumulative 6 day root system', figure, 'rhizodeposits =', summed_quantities6days[0])
        print('cumulative 6 day root system', figure, 'no rhizodeposits =', summed_quantities6days[1])
        print('cumulative 15 day root system', figure, 'rhizodeposits =', summed_quantities15days[0])
        print('cumulative 15 day root system', figure, 'no rhizodeposits =', summed_quantities15days[1])
        print('cumulative 30 day root system', figure, 'rhizodeposits =', summed_quantities30days[0])
        print('cumulative 30 day root system', figure, 'no rhizodeposits =', summed_quantities30days[1])
    
        # print('Max rate 6 day root system', figure, 'rhizodeposits =', np.amax(data_rhizodeposits6days))
        # print('Min rate 6 day root system', figure, 'rhizodeposits =', np.amin(data_rhizodeposits6days))
        # print('Max rate 6 day root system', figure, 'no rhizodeposits =', np.amax(data_no_rhizodeposits6days))
        # print('Min rate 6 day root system', figure, 'no rhizodeposits =', np.amin(data_no_rhizodeposits6days))
        # print('Max rate 15 day root system', figure, 'rhizodeposits =', np.amax(data_rhizodeposits15days))
        # print('Min rate 15 day root system', figure, 'rhizodeposits =', np.amin(data_rhizodeposits15days))
        # print('Max rate 15 day root system', figure, 'no rhizodeposits =', np.amax(data_no_rhizodeposits15days))
        # print('Min rate 15 day root system', figure, 'no rhizodeposits =', np.amin(data_no_rhizodeposits15days))
        # print('Max rate 30 day root system', figure, 'rhizodeposits =', np.amax(data_rhizodeposits30days))
        # print('Min rate 30 day root system', figure, 'rhizodeposits =', np.amin(data_rhizodeposits30days))
        # print('Max rate 30 day root system', figure, 'no rhizodeposits =', np.amax(data_no_rhizodeposits30days))
        # print('Min rate 30 day root system', figure, 'no rhizodeposits =', np.amin(data_no_rhizodeposits30days))
        
    else:
        print('cumulative 6 day root system', figure, 'rhizodeposits =', summed_quantities6days[0])
        print('cumulative 6 day root system', figure, 'no rhizodeposits =', summed_quantities6days[1])
        print('cumulative 15 day root system', figure, 'rhizodeposits =', summed_quantities15days[0])
        print('cumulative 15 day root system', figure, 'no rhizodeposits =', summed_quantities15days[1])
        print('cumulative 30 day root system', figure, 'rhizodeposits =', summed_quantities30days[0])
        print('cumulative 30 day root system', figure, 'no rhizodeposits =', summed_quantities30days[1])
        
    
    return summed_quantities

summed_quantities = post_processor('uptake',
                                   'all', 
                                   'sandy_loam', 
                                   'wetting', 
                                   0.069, 
                                   100, 
                                   3000, 
                                   0.12, 
                                   1, 
                                   3, 
                                   10, 
                                   'eq', 
                                   'eq',
                                   1.0)


