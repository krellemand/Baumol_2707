import numpy as np
import numba as nb
import scipy.optimize as optimize
import quantecon 

from consav.linear_interp import interp_1d_vec

@nb.njit(parallel=True)   

def solve_hh_backwards(par,z_trans,r,wt,vbeg_a_plus,vbeg_a,a,c, ell, l, G, inc, u, mu):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """



    # 1. Preparing FOC  

    # a. objective function non binding contraint
    if par.sigma != 1:
        # @nb.njit
        def obj(c, G, par, i_fix, i_z, i_a, vbeg_a_plus):       

            return par.beta * vbeg_a_plus[i_fix, i_z, i_a] - par.alpha**(1/par.sigma)*c**(-1/par.sigma)   * (par.alpha**(1/par.sigma)*c**((par.sigma-1)/par.sigma) + (1-par.alpha)**(1/par.sigma)*G**((par.sigma-1)/par.sigma))**((1-par.sigma*par.rho)/par.sigma-1)
    else:
        def obj(c, G, par, i_fix, i_z, i_a, vbeg_a_plus):       
            print('sigma = 1 not implemented')  


    # b. objective function when constraint is binding
    def obj_ell(elli, G, par, i_fix, i_z, i_a, vbeg_a_plus):

        c_ = (1+r)*par.a_grid[i_a] + wt*elli*z 

        MU_c = par.alpha**(1/par.sigma)*c_**(-1/par.sigma)   * (par.alpha**(1/par.sigma)*c_**((par.sigma-1)/par.sigma) + (1-par.alpha)**(1/par.sigma)*G**((par.sigma-1)/par.sigma))**((1-par.sigma*par.rho)/par.sigma-1)
        MU_ell = par.varphi * elli**(par.nu)

        error = wt*z*MU_c - MU_ell 

        return error


    for i_fix in nb.prange(par.Nfix):

        # a. solve step
        for i_z in nb.prange(par.Nz):

            # i. Prepare 
            z = par.z_grid[i_z] 
            fac =  ((wt*z)/par.varphi)**(1/par.nu) 
            

                # oo. Solutions for c 
            c_endo = np.empty( par.Na)  
            for i_a in nb.prange( par.Na): 
                try:
                    c_endo[i_a] = optimize.root_scalar(lambda c: obj(c, G, par, i_fix, i_z, i_a, vbeg_a_plus), bracket=(1e-16, 700)).root
                    
                    # c_endo[i_a] = quantecon.optimize.root_finding.bisect(obj, 0.1, 40, args=(G, par, i_fix, i_z, i_a, vbeg_a_plus), xtol=2e-12, rtol=8.881784197001252e-16, maxiter=100, disp=True).root

                    # c_endo[i_a] = quantecon.optimize.root_finding.bisect(f, a, b, args=(), xtol=2e-12, rtol=8.881784197001252e-16, maxiter=100, disp=True)
                                
                except:
                    # print('No solution for c')
                    break
            

            # iii. FOC for labor supply
            ell_endo = fac * vbeg_a_plus[i_fix, i_z, :]**(1/par.nu)  

            l_endo = ell_endo* z


            # iv. re-interpolate


            m_endo = c_endo + par.a_grid - wt * l_endo 
            m_exo = (1+r)*par.a_grid


            # v. interpolation to find c and l using endogenous grid, on exogenous grid
            
            interp_1d_vec(m_endo,c_endo,m_exo,c[i_fix,i_z,:])

            # Interpolating labor supply
            interp_1d_vec(m_endo,ell_endo,m_exo,ell[i_fix,i_z,:])    # Grid 1 = m_endo, Value 1 = ell_endo,  Input vector = m_exo, Output vector = ell
            l[i_fix,i_z,:] = ell[i_fix,i_z,:]*z            
            

            # vi. saving
            a[i_fix,i_z,:] = m_exo + wt*l[i_fix,i_z,:] - c[i_fix,i_z,:] #- tax_
            # print('a', a[i_fix,i_z,:])

            # vii. refinement at constraint
            for i_a in range(par.Na):


                if a[i_fix,i_z,i_a] < 1e-8:
                    
                    # o. binding constraint for a
                    # print(f'binding constraint for {i_fix} {i_z} {i_a}')
                    a[i_fix,i_z,i_a] = 0.0
                    # print(f'Ell = {ell[i_fix,i_z,i_a]:.02f}')
  
                    # oo. FOC for labor supply / obj function 
                    # def obj_ell(elli, G, par, i_fix, i_z, i_a, vbeg_a_plus):

                    #     c_ = (1+r)*par.a_grid[i_a] + wt*elli*z #- tax_ 
                    #     # print('a_', par.a_grid[i_a])
                    #     # print(f'wt = {wt:.02f}')
                    #     # print(f'z = {z:.02f}')
                    #     # print(f'elli = {elli:.02f}')
                    #     # print(f'tax = {tax_:.02f}')
                    #     # print(f'c_ = {c_:.02f}')
                    #     MU_c = par.alpha**(1/par.sigma)*c_**(-1/par.sigma)   * (par.alpha**(1/par.sigma)*c_**((par.sigma-1)/par.sigma) + (1-par.alpha)**(1/par.sigma)*G**((par.sigma-1)/par.sigma))**((1-par.sigma*par.rho)/par.sigma-1)
                    #     MU_ell = par.varphi * elli**(par.nu)

                    #     # print(f'MU_c = {MU_c:.02f}')
                    #     # print(f'MU_ell = {MU_ell:.02f}')

                    #     error = wt*z*MU_c - MU_ell # **** Skal w og z ikke ganges på den MU_ell
                    #     # print(f'error = {error:.09f}')
                    #     return error
                
                    try: 
                        # ooo. Finding ell consistent with marginal utility of c = marginal utility of labor given a = 0
                        # ell_ = optimize.root_scalar(lambda elli: obj_ell(elli, G, par, i_fix, i_z, i_a, vbeg_a_plus), bracket=(0.2, 4)).root
                        ell_ = optimize.root_scalar(lambda elli: obj_ell(elli, G, par, i_fix, i_z, i_a, vbeg_a_plus), bracket=(0.2, 4)).root
                        
                        
                        # ell_ = quantecon.optimize.root_finding.bisect(obj_ell, 0.1, 40, args=( G, par, i_fix, i_z, i_a, vbeg_a_plus), xtol=2e-12, rtol=8.881784197001252e-16, maxiter=100, disp=True).root
                      
                      
                       # print(f'ell_ = {ell_:.02f}')
                        # oooo. save
                        c[i_fix,i_z,i_a] = (1+r)*par.a_grid[i_a] + wt*ell_*z #- tax_ 
                        ell[i_fix,i_z,i_a] = ell_
                        l[i_fix,i_z,i_a] = ell_*z

                        # h = (par.alpha**(1/par.sigma)*c[i_fix]**(par.sigma-1/par.sigma) + (1-par.alpha)**(1/par.sigma)*G**(par.sigma-1/par.sigma))**(par.sigma/(par.sigma-1))
                        # u[i_fix,i_z,i_a] = ((h)**(1-par.rho))/1-par.rho - par.varphi*(ell[i_fix]**(1+par.nu))/(1+par.nu) 

                        # print(f'Z =  {z:.02f} a = 0 c=  {c[i_fix, i_z,0]:.02f},  ell =  {ell[i_fix, i_z, 0]:.02f}')
                        # print(f'For z =  {z:.02f},  {a[i_fix,i_z,i_a]:.02f}, c=  {c[i_fix, i_z,i_a]:.02f},  ell = {ell[i_fix, i_z, i_a]:.02f}')
                    
                    except:
                        print('No solution for ell')

                # print(f'For z =  {z:.02f}, a=  {a[i_fix,i_z,i_a]:.02f}, c=  {c[i_fix, i_z,i_a]:.02f},  ell = {ell[i_fix, i_z, i_a]:.02f}')

                if a[i_fix,i_z,i_a] > 800:
                    print('High assets')

                else:
                    break 
            # print(f'For z =  {z:.02f}, a=  {a[i_fix,i_z,i_a]:.02f}, c=  {c[i_fix, i_z,i_a]:.07f},  ell = {ell[i_fix, i_z, i_a]:.02f}')





             # viii. utility Test***
        
        
        u[i_fix,:, :] = (((par.alpha**(1/par.sigma) * c[i_fix]**((par.sigma-1)/par.sigma) + (1-par.alpha)**(1/par.sigma) * G**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1)))**(1-par.rho) / (1-par.rho)) - (par.varphi * (ell[i_fix]**(1+par.nu)) / (1+par.nu))


        
        inc[i_fix] = wt*l[i_fix] + r*par.a_grid  #- tax_
        mu[i_fix,:,:] = 0 #par.alpha**(1/par.sigma)*c**(-1/par.sigma)   * (par.alpha**(1/par.sigma)*c**((par.sigma-1)/par.sigma) + (1-par.alpha)**(1/par.sigma)*G**((par.sigma-1)/par.sigma))**((1-par.sigma*par.rho)/par.sigma-1)




        # h[i_fix,:,:] = (par.alpha**(1/par.sigma)*c[i_fix]**(par.sigma-1/par.sigma) + (1-par.alpha)**(1/par.sigma)*G**(par.sigma-1/par.sigma))**(par.sigma/(par.sigma-1))
                                                                                                                                      

        # u[i_fix,:,:] = ((h[i_fix,:,:])**(1-par.rho))/1-par.rho - par.varphi*(ell[i_fix]**(1+par.nu))/(1+par.nu) 
        # u[i_fix,:,:] = (    ((par.alpha**(1/par.sigma)*c[i_fix]**((par.sigma-1)/par.sigma) + (1-par.alpha)**(1/par.sigma)*G**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))   )**(1-par.rho))/1-par.rho    -   par.varphi*(ell[i_fix]**(1+par.nu))/(1+par.nu) 

        # b. expectation step
        v_a = (1+r) * par.alpha**(1/par.sigma)*c[i_fix]**(-1/par.sigma)   * (par.alpha**(1/par.sigma)*c[i_fix]**((par.sigma-1)/par.sigma) + (1-par.alpha)**(1/par.sigma)*G**((par.sigma-1)/par.sigma))**((1-par.sigma*par.rho)/par.sigma-1) # Marginal Value Function
        # print(f'sum v_a: {np.sum(v_a)}')
        vbeg_a[i_fix] = z_trans[i_fix]@v_a 
        # print(f'sum v_beg a: {np.sum(vbeg_a[i_fix])}')

        # print(np.sum(vbeg_a[i_fix]))



