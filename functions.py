import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']



#-------------------------------------------------------------------------------------------------------------
#                       Additional functions
#-------------------------------------------------------------------------------------------------------------


def create_model_dict (model, list_tfp, list_G, do_print=False):
    """Create a dictionary of models with different TFP and G
    Args:
    model: Baseline model
    list_tfp (list): List of TFP values
    list_G (list): List of G values
    Returns:
    Dictionary of models with different TFP and G values"""

    models_all = {}

    for TFP in list_tfp:


        print(f'Solving for TFP = {TFP}')
        model.par.Gamma_Y_ = TFP

        models = {}

        for G in list_G:

            model_ = model.copy(name=f'TFP = {TFP} G = {G}')
            model_.par.G_ = G

            try:
                model_.find_ss(do_print=do_print)
                # model_.test_ss()
                models[f'G = {G}'] = model_
                print(f'solved for TFP = {TFP} G = {G}')

            except:
                print(f'TFP = {TFP} G = {G} did not converge')
        
        models_all[f'TFP = {TFP}'] = models
    models_all['name'] = model.name
    
    return models_all



#-------------------------------------------------------------------------------------------------------------
#                       Figures
#-------------------------------------------------------------------------------------------------------------


def plot_policy(model):
    """Plot the policy functions of the model"""
    ss = model.ss
    par = model.par
    path = model.path

    i_fix = 0

    fig = plt.figure(figsize=(18,4),dpi=100)
    a_max = 100
    # Add an overall title/heading
    fig.suptitle(f'Policy function model: {model.name}', fontsize=16)


    # a. consumption
    I = par.a_grid < a_max

    ax = fig.add_subplot(1,3,1)
    ax.set_title(f'consumption')

    for i_z in [0,par.Nz//2,par.Nz-1]:
        ax.plot(par.a_grid[I],ss.c[i_fix,i_z,I],label=f'i_z = {i_z}')

    ax.legend(frameon=True)
    ax.set_xlabel('savings, $a_{t-1}$')
    ax.set_ylabel('consumption, $c_t$')

    # b. saving
    I = par.a_grid < a_max

    ax = fig.add_subplot(1,3,2)
    ax.set_title(f'saving')

    for i_z in [0,par.Nz//2,par.Nz-1]:
        ax.plot(par.a_grid[I],ss.a[i_fix,i_z,I],label=f'i_z = {i_z}')

    ax.set_xlabel('savings, $a_{t-1}$')
    ax.set_ylabel('savings, $a_{t}$')

    # c. labor supply
    I = par.a_grid < a_max

    ax = fig.add_subplot(1,3,3)
    ax.set_title(f'labor_supply')

    for i_z in [0,par.Nz//2,par.Nz-1]:
        ax.plot(par.a_grid[I],ss.ell[i_fix,i_z,I],label=f'i_z = {i_z}')

    ax.set_xlabel('savings, $a_{t-1}$')
    ax.set_ylabel('labor supply, $n_{t}$')
    ax.set_ylim([0,3.5])
    fig.tight_layout()

    return fig


def plot_cum(models, varnames, ncols= 3, xlim= [], print_gini = False, title= None): 
    

    """Plot the cumulative distribution function of a variable in the model
    Args:
    var (str): Variable name
    models: List of models to plot (eg. varying G or varying TFP)
    xlim (list): x-axis limits
    Returns:
    CDF plot and Gini coefficient
    """

    num = len(varnames)
    nrows = num//ncols+1
    if num%ncols == 0: nrows -= 1
    
    fig = plt.figure(figsize=(6*ncols,4*nrows),dpi=100)
    if title != None:
        fig.suptitle(title, fontsize=16)

    for i,varname in enumerate(varnames):
        var = varname
        
        ax = fig.add_subplot(nrows,ncols,i+1)
        title = varname
        ax.set_title(title,fontsize=14)

    #var = 'a'
    # Flattening  data og weits

        for model in models:
            try:
                model = model
                var_ = model.ss.__dict__[var][:,:,:].flatten()
                weight = model.ss.D[:,:,:].flatten()

                # Sorting data and weits
                sorted_var, sorted_weights = zip(*sorted(zip(var_, weight)))
                sorted_var = np.array(sorted_var)
                sorted_weights = np.array(sorted_weights)

                # Calculating the cumulative sum of sorted weights
                cumulative = np.cumsum(sorted_weights)

                ax.plot(sorted_var, cumulative, label=model.name ) #, color = model.c)
                ax.set_xscale('symlog')


                # Normalized cumulative weights (needed for Lorenz curve)
                normalized_cumulative_weights = cumulative / cumulative[-1]

                # Normalized weighted cumulative values
                normalized_weighted_cumulative_values = np.cumsum(sorted_var * sorted_weights) / np.cumsum(sorted_var * sorted_weights)[-1]

                # Gini calculation
                area_under_lorenz_curve = np.trapz(normalized_weighted_cumulative_values, normalized_cumulative_weights)
                weighted_gini = 1 - 2 * area_under_lorenz_curve

                if print_gini:
                    print(f'{model.name} {var}')
                    print(f'Weighted Gini Coefficient: {weighted_gini:.2f}')
            except:
                print(f'Could not plot {var} for {model.name}')


        if xlim != []:
            ax.set_xlim(xlim)

        # ax.set_xlabel(f'{var}')
        ax.legend(loc= 'lower right')
        #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        #ax.set_xscale('symlog')

        # ax.set_ylabel('Cumulative Probability')
        # ax.set_title('Cumulative Distribution Function (CDF) of ' + var)
        ax.grid(True)

    # return fig


def plot_G(models,  varnames, list_TFP, list_G, ncols=3, title=None):
    """Plot the steady state values of a variable in the models over G
    Args:
    varnames (list): List of variable names
    models: Dictionary of models
    list_G (list): List of G values
    list_TFP (list): List of TFP values
    ncols (int): Number of columns in the plot
    title (str): Title of the plot

    Returns:
    Plot of the steady state values of the variables over G with each TFP level plotted"""

    num = len(varnames)
    nrows = num//ncols+1
    if num%ncols == 0: nrows -= 1
    
    fig = plt.figure(figsize=(6*ncols,4*nrows),dpi=100)
    # Setting title 
    if title != None:
        fig.suptitle(title, fontsize=16)



    for i,varname in enumerate(varnames):
        
        ax = fig.add_subplot(nrows,ncols,i+1)
        title = varname
        ax.set_title(title,fontsize=14)

        for TFP in list_TFP:

            list_values = []

            for G in list_G:
                try:
                    value = models[f'TFP = {TFP}'][f'G = {G}'].ss.__dict__[varname]
                    list_values.append(value)
                except:
                    list_values.append(np.nan)



            ax.plot(list_G, list_values, label=f'TFP = {TFP}')
            ax.legend()

def plot_G_compare(models_list,  varnames, list_TFP, list_G, ncols=3, title=None):
    """Plot the steady state values of a variable in the models over G
    Args:
    varnames (list): List of variable names
    models: Dictionary of models
    list_G (list): List of G values
    list_TFP (list): List of TFP values
    ncols (int): Number of columns in the plot
    title (str): Title of the plot

    Returns:
    Plot of the steady state values of the variables over G with each TFP level plotted"""



    num = len(varnames)
    nrows = num//ncols+1
    if num%ncols == 0: nrows -= 1
    
    fig = plt.figure(figsize=(6*ncols,4*nrows),dpi=100)
    # Setting title 
    # if title == None:

    if title != None:
        fig.suptitle(title, fontsize=16)


    for i,varname in enumerate(varnames):
        
        ax = fig.add_subplot(nrows,ncols,i+1)
        title = varname
        ax.set_title(title,fontsize=14)

        for models in models_list:

            for TFP in list_TFP:

                list_values = []

                for G in list_G:
                    try:
                        value = models[f'TFP = {TFP}'][f'G = {G}'].ss.__dict__[varname]
                        list_values.append(value)
                    except:
                        list_values.append(np.nan)

                ax.plot(list_G, list_values, label=f'{models["name"] } TFP = {TFP}')
    ax.legend()

def plot_dec(model, var, decomposition_input, T_max=50):

    """Plot the decomposition of the path of a variable in the model
    Args:
    model: Model object
    var (str): Variable being decomposed 
    decomposition_input (list): List of inputs to decompose over
    T_max (int): Maximum time period to plot

    Returns:
    Plot of the decomposition of the path of the variable"""


    fig = plt.figure(figsize=(6,4),dpi=100)
    ax = fig.add_subplot(1,1,1)
        
    i_color = 0
    # for use_inputs in [ ['wt'], ['Gamma_Y'], ['r'], ['tau'], 'all']:
    # for use_inputs in [ 'G','wt', 'r', 'all']:
    for use_inputs in decomposition_input:
        
        # a. compute
        print(use_inputs)
        path_alt = model.decompose_hh_path(do_print=True,use_inputs=use_inputs)
        print('')
        
        # b. plot
        if use_inputs is None:
            label = 'no inputs'
            ls = '--'
            color = 'black'
        elif use_inputs == 'all':
            label = 'all inputs'
            ls = '-'
            color = 'black'
        else:
            label = f'only effect from {use_inputs[:]}'
            ls = '-'
            color = colors[i_color]
            i_color += 1
            
        ax.plot((path_alt.__dict__[var][:T_max]/model.ss.__dict__[var]-1)*100,ls=ls,color=color,label=label);

    ax.set_ylabel('% diff to s.s.')
    ax.legend(frameon=True,bbox_to_anchor=(1.01,0.99));




#-------------------------------------------------------------------------------------------------------------
#                       Tables 
#-------------------------------------------------------------------------------------------------------------

def table_ss(models):

    def df_(model):
        if model is False:
            return None

        # 1. Saving steady state variables in a DataFrame
        data = {varname: [f'{model.ss.__dict__[varname]:.3f}'] for varname in model.varlist}
        df = pd.DataFrame(data).T  # Transpose to get variables as rows
        df.columns = [model.name]
        df.index.name = 'Variable'

        # 2. Additional information 

        # a. Government share of the economy
        ratio = model.ss.G / model.ss.Y
        df.loc['G/Y'] = [f'{ratio:.3f}']

        # b. Utility
        tot_util = model.calc_u()
        df.loc['Discounted utility'] = [f'{tot_util:.3f}']

        return df

    # Concatenating the DataFrames for all models
    dataframes = [df_(model) for model in models if model]
    table = pd.concat(dataframes, axis=1)

    return table


def table_summary(models):

    par_rel = ['sigma', 'rho', 'alpha',  'nu', 'varphi',  'Gamma_Y_', 'G_']

    var_rel = ['U_hh','Y', 'G', 'L_Y', 'L_G', 'w', 'wt', 'tau', 'K']


    def df_(model):

 
        if model == False:
            return None
        
        # 1. Saving steady state variables in a DataFrame

        data = {varname: [f'{model.ss.__dict__[varname]:.3f}'] for varname in var_rel}
        df = pd.DataFrame(data).T  # Transpose to get variables as rows
        df.columns = [model.name]
        df.index.name = 'Variable'

        # 2. Additional information 

        # a. Government share of the economy
        ratio = model.ss.G/model.ss.Y
        df.loc['G/Y'] = [f'{ratio:.3f}']


        # b. Utility
        tot_util = model.calc_u()
        df.loc['Discounted utility'] = [f'{tot_util:.3f}']


        for par_ in par_rel:
            df.loc[par_] = [f'{model.par.__dict__[par_]:.1f}']

        return df

    # 3. Concatenating the DataFrames
    dataframes = [df_(model) for model in models if model]
    table = pd.concat(dataframes, axis=1)


    return table




#-------------------------------------------------------------------------------------------------------------
#                       Solvable models test
#-------------------------------------------------------------------------------------------------------------

def solvable_par(model, are_you_sure=False):

    if are_you_sure:

        par = model.par
        ss = model.ss


        # Initialize an empty DataFrame
        columns = [ 'solved', 'elapsed_time', 'sigma', 'alpha', 'G', 'nu', 'varphi',  'r', 'w', 'A_hh', 'L_hh', 'C_hh', 'rho', 'beta']
        results_df = pd.DataFrame(columns=columns)

        sigma_list = [0.7, 0.9, 1.1]
        alpha_list = [0.5, 0.7, 0.8]
        G_list = [0.4, 0.5, 0.7]
        nu_list = [1.0, 1.5, 2.0]
        varphi_list = [0.5, 0.8, 1.0]
        rho_list = [1.0, 2.0]


        for sigma in sigma_list:
            par.sigma = sigma
            for alpha in alpha_list:
                par.alpha = alpha
                for G in G_list:
                    par.G_ = ss.G = G
                    for nu in nu_list:
                        par.nu = nu
                        for varphi in varphi_list:
                            par.varphi = varphi
                            for rho in rho_list:
                                par.rho = rho

                                
                                # Start timing
                                t0 = time.time()
                                try:
                                    #print(f'Running with sigma={sigma}, alpha={alpha}, G={G}, nu={nu}, varphi={varphi}, rho={rho}')
                                    
                                    model.find_ss()  # assuming these methods set attributes on the model or par
                                    model.test_ss()
                                    solved = True
                                    print('Solved')
                                        # End timing
                                except:
                                    # End timing
                                    solved = False
                                    print('Not solved')

                                elapsed_time = time.time() - t0

                                # Append results as a new row in the DataFrame
                                result_row = {
                                    'sigma': sigma, 'alpha': alpha, 'G': G, 'nu': nu, 'varphi': varphi, 'solved': solved, 
                                    'elapsed_time': elapsed_time, 'r': ss.r, 'w': ss.w, 'A_hh': ss.A_hh, 'L_hh': ss.L_hh, 'C_hh': ss.C_hh, 
                                    'rho': rho, 
                                }
                                results_df = results_df.append(result_row, ignore_index=True)

        # Export the DataFrame to Excel
        try: 
            results_df.to_excel('tabs/results_df2.xlsx')
        except:
            print('Could not export to Excel')
        return results_df
    
    else:
        print('Way too slow, dont run this')



#-------------------------------------------------------------------------------------------------------------
#                       Old 
#-------------------------------------------------------------------------------------------------------------


def plot_cum_old(var, model, model2= False, model3=False, model4=False, xlim = []):

    """Plot the cumulative distribution function of a variable in the model
    Args:
    var (str): Variable name
    models: Model object
    xlim (list): x-axis limits
    Returns:
    CDF plot and Gini coefficient
    """


    fig = plt.figure(figsize=(12,4),dpi=100)
    ax = fig.add_subplot(1,2,1)
    ax.set_title('')

    #var = 'a'
    # Flattening  data og weits
    var_ = model.ss.__dict__[var][:,:,:].flatten()
    weight = model.ss.D[:,:,:].flatten()

    # Sorting data and weits
    sorted_var, sorted_weights = zip(*sorted(zip(var_, weight)))
    sorted_var = np.array(sorted_var)
    sorted_weights = np.array(sorted_weights)

    # Calculating the cumulative sum of sorted weights
    cumulative = np.cumsum(sorted_weights)

    ax.plot(sorted_var, cumulative, label=model.name, color = model.c)
    ax.set_xscale('symlog')


    # Normalized cumulative weights (needed for Lorenz curve)
    normalized_cumulative_weights = cumulative / cumulative[-1]

    # Normalized weighted cumulative values
    normalized_weighted_cumulative_values = np.cumsum(sorted_var * sorted_weights) / np.cumsum(sorted_var * sorted_weights)[-1]

    # Gini calculation
    area_under_lorenz_curve = np.trapz(normalized_weighted_cumulative_values, normalized_cumulative_weights)
    weighted_gini = 1 - 2 * area_under_lorenz_curve

    # print(f'{model.name} {var}')
    # print(f'Weighted Gini Coefficient: {weighted_gini:.2f}')




    if model2 != False:
        model = model2
        
        var_ = model.ss.__dict__[var][:,:,:].flatten()
        weight = model.ss.D[:,:,:].flatten()

        # Sorting data and weits
        sorted_var, sorted_weights = zip(*sorted(zip(var_, weight)))
        sorted_var = np.array(sorted_var)
        sorted_weights = np.array(sorted_weights)

        # Calculating the cumulative sum of sorted weights
        cumulative = np.cumsum(sorted_weights)

        ax.plot(sorted_var, cumulative, label=model.name, color = model.c)
        ax.set_xscale('symlog')


        # Normalized cumulative weights (needed for Lorenz curve)
        normalized_cumulative_weights = cumulative / cumulative[-1]

        # Normalized weighted cumulative values
        normalized_weighted_cumulative_values = np.cumsum(sorted_var * sorted_weights) / np.cumsum(sorted_var * sorted_weights)[-1]

        # Gini calculation
        area_under_lorenz_curve = np.trapz(normalized_weighted_cumulative_values, normalized_cumulative_weights)
        weighted_gini = 1 - 2 * area_under_lorenz_curve


        # print(f'{model.name} {var}')
        # print(f'Weighted Gini Coefficient: {weighted_gini:.2f}')


    if model3 != False:
        model = model3
        
        var_ = model.ss.__dict__[var][:,:,:].flatten()
        weight = model.ss.D[:,:,:].flatten()

        # Sorting data and weits
        sorted_var, sorted_weights = zip(*sorted(zip(var_, weight)))
        sorted_var = np.array(sorted_var)
        sorted_weights = np.array(sorted_weights)

        # Calculating the cumulative sum of sorted weights
        cumulative = np.cumsum(sorted_weights)

        ax.plot(sorted_var, cumulative, label=model.name, color = model.c)
        ax.set_xscale('symlog')


        # Normalized cumulative weights (needed for Lorenz curve)
        normalized_cumulative_weights = cumulative / cumulative[-1]

        # Normalized weighted cumulative values
        normalized_weighted_cumulative_values = np.cumsum(sorted_var * sorted_weights) / np.cumsum(sorted_var * sorted_weights)[-1]

        # Gini calculation
        area_under_lorenz_curve = np.trapz(normalized_weighted_cumulative_values, normalized_cumulative_weights)
        weighted_gini = 1 - 2 * area_under_lorenz_curve


        print(f'{model.name} {var}')
        print(f'Weighted Gini Coefficient: {weighted_gini:.2f}')



    if model4 != False:
        var_ = model.ss.__dict__[var][:,:,:].flatten()
        weight = model.ss.D[:,:,:].flatten()

        # Sorting data and weits
        sorted_var, sorted_weights = zip(*sorted(zip(var_, weight)))
        sorted_var = np.array(sorted_var)
        sorted_weights = np.array(sorted_weights)

        # Calculating the cumulative sum of sorted weights
        cumulative = np.cumsum(sorted_weights)

        ax.plot(sorted_var, cumulative, label=model.name, color = model.c)
        ax.set_xscale('symlog')


        # Normalized cumulative weights (needed for Lorenz curve)
        normalized_cumulative_weights = cumulative / cumulative[-1]

        # Normalized weighted cumulative values
        normalized_weighted_cumulative_values = np.cumsum(sorted_var * sorted_weights) / np.cumsum(sorted_var * sorted_weights)[-1]

        # Gini calculation
        area_under_lorenz_curve = np.trapz(normalized_weighted_cumulative_values, normalized_cumulative_weights)
        weighted_gini = 1 - 2 * area_under_lorenz_curve


        # print(f'{model.name} {var}')
        # print(f'Weighted Gini Coefficient: {weighted_gini:.2f}')



    if xlim != []:
        ax.set_xlim(xlim)

    ax.set_xlabel(f'{var}')
    ax.legend(loc= 'upper left')
    #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    #ax.set_xscale('symlog')

    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function (CDF) of ' + var)
    ax.grid(True)

    # return fig




def table_summary_old(model_1, model_2=False, model_3=False, model_4=False, model_5 = False):

    par_rel = ['sigma', 'rho', 'alpha',  'nu', 'varphi',  'Gamma_Y_', 'G_']

    var_rel = ['U_hh','Y', 'G', 'L_Y', 'L_G', 'w', 'wt', 'tau', 'K']


    def df_(model):

 
        if model == False:
            return None
        
        # 1. Saving steady state variables in a DataFrame

        data = {varname: [f'{model.ss.__dict__[varname]:.3f}'] for varname in var_rel}
        df = pd.DataFrame(data).T  # Transpose to get variables as rows
        df.columns = [model.name]
        df.index.name = 'Variable'

        # 2. Additional information 

        # a. Government share of the economy
        ratio = model.ss.G/model.ss.Y
        df.loc['G/Y'] = [f'{ratio:.3f}']


        # b. Utility
        tot_util = model.calc_u()
        df.loc['Discounted utility'] = [f'{tot_util:.3f}']


        for par_ in par_rel:
            df.loc[par_] = [f'{model.par.__dict__[par_]:.1f}']

        return df

    # 3. Concatenating the DataFrames

    table = pd.concat([df_(model_1), df_(model_2), df_(model_3), df_(model_4), df_(model_5)], axis=1)


    return table



def table_ss_old(model_1, model_2=False, model_3=False, model_4=False, model_5 = False):


    def df_(model):

        if model == False:
            return None
        
        # 1. Saving steady state variables in a DataFrame

        data = {varname: [f'{model.ss.__dict__[varname]:.3f}'] for varname in model.varlist}
        df = pd.DataFrame(data).T  # Transpose to get variables as rows
        df.columns = [model.name]
        df.index.name = 'Variable'

        # 2. Additional information 

        # a. Government share of the economy
        ratio = model.ss.G/model.ss.Y
        df.loc['G/Y'] = [f'{ratio:.3f}']

        # b. Utility
        tot_util = model.calc_u()
        df.loc['Discounted utility'] = [f'{tot_util:.3f}']

        return df

    # 3. Concatenating the DataFrames
    table = pd.concat([df_(model_1), df_(model_2), df_(model_3), df_(model_4), df_(model_5)], axis=1)


    return table


