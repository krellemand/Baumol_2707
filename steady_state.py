import time
import numpy as np

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst
from consav.misc import elapsed
import scipy.optimize as optimize
import quantecon.optimize.root_finding as root_finding_2
import numpy as np
import numba as nb

import root_finding

def prepare_hh_ss(model):
    """ prepare the household block to solve for steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############
    
    # a. eta grid - Time-invariant productivity

    # par.eta_grid[:] = np.array([1,1.1])
    # par.eta_grid[:] = np.array([1.0])


    # b. iota grid - typespecific lump sum tax
    # par.iota_grid[:] = np.array([0.0,0.0])
    # par.iota_grid[:] = np.array([1.0])

    # c. a
    par.a_grid[:] = equilogspace(0.0,ss.wt*par.a_max,par.Na)
    
    # d. z
    par.z_grid[:],z_trans,z_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,par.Nz)



    #############################################
    # 2. transition matrix initial distribution #
    #############################################
    
    # note: the guess here is somewhat arbitrary

    for i_fix in range(par.Nfix):
        ss.z_trans[i_fix,:,:] = z_trans
        ss.Dbeg[i_fix,:,0] = z_ergodic/par.Nfix # ergodic at a_lag = 0.0
        ss.Dbeg[i_fix,:,1:] = 0.0 # none with a_lag > 0.0

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    # note: the guess here is somewhat arbitrary
    
    # a. raw value
    y = ss.wt*par.z_grid
    c = m = (1+ss.r)*par.a_grid[np.newaxis,:] + y[:,np.newaxis]
    #v_a = (1+ss.r)*c**(-par.sigma)
    v_a = (1+ss.r) * par.alpha**(1/par.sigma)*c**(-1/par.sigma)   * (par.alpha**(1/par.sigma)*c**((par.sigma-1)/par.sigma) + (1-par.alpha)**(1/par.sigma)*ss.G**((par.sigma-1)/par.sigma))**((1-par.sigma*par.rho)/par.sigma-1)

    # b. expectation
    ss.vbeg_a[:] = ss.z_trans@v_a

# @nb.njit
def obj_ss(x,model,do_print=False):
    """ evaluate steady state"""

    par = model.par
    ss = model.ss


    # a. production
    KL = x[0] # Guessing on capital  
    tau =  x[1] # Guessing on tau

    # Productivity
    ss.Gamma_Y = par.Gamma_Y_
    ss.Gamma_G = par.Gamma_G_


    # a. firms
    ss.rK = par.kappa*ss.Gamma_Y*(KL)**(par.kappa-1)
    ss.w = (1.0-par.kappa)*ss.Gamma_Y*(KL)**par.kappa

    ss.r = ss.rK - par.delta
    ss.wt = (1-tau)*ss.w

    # government 
    ss.G = par.G_
    ss.L_G = ss.G/ss.Gamma_G
    # ss.Chi = par.Chi_


    # d. households HouseholdProblem solved from wt, r and chi  
    # print(f'Solving HH for r = {ss.r:6.3f} and wt = {ss.wt:6.3f} and G = {ss.G}, Chi = {ss.Chi}')
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)
    # print(f'HH L = {ss.L_hh:6.3f} and A = {ss.A_hh:6.3f} and C = {ss.C_hh:6.3f} ')

    # Market clearing
    ss.L = ss.L_hh                     #*** Clearing_L clearer mekanisk  -  Ud fra det her clearer markederne rent mekanisk, er clearing_L overflødig, er den her overflødig? Eller beregner vi den her for at bruge og så det bare pænt at have begge steder
    ss.L_Y = ss.L - ss.L_G
    ss.K = KL*ss.L_Y

    # Additional variables
    G_inc = tau*ss.w*ss.L #+ ss.Chi
    G_exp = ss.L_G*ss.w 
    ss.G_exp = G_exp

    ss.tau = tau

    ss.Y = ss.Gamma_Y*(ss.K+1e-12)**(par.kappa)*(ss.L_Y+1e-12)**(1-par.kappa) # Total privat production
    ss.I = par.delta*ss.K
    ss.A = ss.K        

    ss.G_share = ss.G/ss.Y
    ss.G_exp_share = G_exp/(G_exp + ss.L_Y*ss.w)


    ss.clearing_A = ss.A - ss.A_hh     
    ss.clearing_L = ss.L - ss.L_hh
    ss.clearing_Y = ss.Y - (ss.C_hh + ss.I )    
    ss.clearing_G = G_inc - G_exp # Gov budget constraint 


    # print(f'{ss.K = :6.3f}')
    # print(f'{ss.L = :6.3f}')
    # print(f'{ss.L_Y = :6.3f}')
    # print(f'{ss.L_G = :6.3f}')
    # print(f'{ss.Y = :6.3f}')
    # print(f'{ss.rK = :6.3f}')
    # print(f'{ss.wt = :6.3f}')
    # print(f'tau = {ss.tau:6.3f}')

    # return  ss.clearing_L, ss.clearing_G
    return  ss.clearing_A , ss.clearing_G


def find_ss(model,do_print=False):
    """ find the steady state """

    t0 = time.time()

    par = model.par
    ss = model.ss

    KL_min = ((1/par.beta+par.delta-1)/(par.kappa*par.Gamma_Y_))**(1/(par.kappa-1)) + 1e-2
    KL_max = (par.delta/(par.kappa*par.Gamma_Y_))**(1/(par.kappa-1))-1e-2 # Hvorfor? 
    KL_mid = (KL_min+KL_max)/2 # middle point between max values as initial capital labor ratio


    # a. solve for K and L
    #initial_guess =  (3.0, 0.25) 
    initial_guess =  (KL_mid , 0.25) 

    if do_print: print(f'starting at [{initial_guess[0]:.4f}]')

    res = optimize.root(obj_ss, initial_guess, args=(model,)) #*** Correct
#    res = root_finding_2.newton_secant(obj_ss, initial_guess, args=(model,), tol=1.48e-08, maxiter=50, disp=True)
    # res = root_finding_2.bisect(obj_ss, KL_min, KL_max, args=(model,), xtol=1.48e-08, maxiter=50, disp=True)
    if do_print: 
        print('')
        print(res)
        print('')
    
    # b. final evaluations
    obj_ss(res.x,model)

    # c. show
    if do_print:

        print(f'steady state found for {model.name}in {elapsed(t0)}')
        print(f'{ss.K = :6.3f}')

        print(f'{ss.A_hh = :6.3f}')
        print(f'{ss.L = :6.3f}')
        print(f'{ss.Y = :6.3f}')
        print(f'{ss.r = :6.3f}')
        print(f'{ss.w = :6.3f}')
        print(f'{ss.clearing_A = :.2e}')
        print(f'{ss.clearing_L = :.2e}')
        print(f'{ss.clearing_Y = :.2e}')
    



