import numpy as np

from EconModel import EconModelClass
from GEModelTools import GEModelClass

import steady_state
import household_problem

class HANCModelClass(EconModelClass,GEModelClass):    

    # remember in model = EconModelClass(name='') we call:
    # self.settings()ba
    # self.setup()
    # self.allocate()

    def settings(self):
        """ fundamental settings """

        # a. namespaces (typically not changed)
        self.namespaces = ['par','ini','ss','path','sim'] # not used today: 'ini', 'path', 'sim'

        # settings required for in GEModelClass
        # important for allocate_GE in self.allocate()

        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['r','wt', 'G'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs (not used today)
        self.outputs_hh = ['a','c','l', 'ell', 'inc', 'u', 'mu'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['G', 'Gamma_Y', 'Gamma_G'] # exogenous shocks (not used today)
        self.unknowns = ['K', 'L_Y', 'tau'] # endogenous unknowns (not used today)
        self.targets = ['clearing_A', 'clearing_Y', 'clearing_G' ]#, 'clearing_G'] # targets = 0self.targets = [] # targets = 0 (not used today)
        self.blocks = [ # list of strings to block-functions
            'blocks.production_firm',
            'blocks.mutual_fund',
            'blocks.government',
            'hh', # household block
            'blocks.market_clearing', 
            'blocks.additional_var'
            ]
        

        
        # d. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards

    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nfix = 1 # number of fixed discrete states (none here)
        par.Nz = 7 # number of stochastic discrete states (here productivity)

        # a. preferences
        par.sigma = 0.6 # Consumption substitution elasticity (Intra-temporal elasticity of substitution)
        par.rho = 2.0 # Inverse intertemporal elasticity of substitution
        par.alpha = 0.60 # weight on private consumption in utility function
        par.beta = 0.975 # discount factor
        par.nu = 2.0 #inverse Frisch elasticity of labor supply
        par.varphi = 1.0 # disutility of labor  


        # b. income parameters
        par.rho_z = 0.96 # AR(1) parameter
        par.sigma_psi = 0.15 # std. of shock
        # par.rho_z = 0.95 # AR(1) parameter
        # par.sigma_psi = 0.30*(1.0-par.rho_z**2.0)**0.5 # std. of persistent shock

        # c. production and investment
        par.kappa = 0.3 # coubdouglas production function parameter
        par.delta = 0.10 # depreciation rate
        par.Gamma_ss = 1.0 # direct approach: technology level in steady state

        # d. government
        par.Gamma_Y_ = 1.0
        par.Gamma_G_ = 1.0
        par.G_ = 0.3 # government consumption
        # par.Chi_ = 1e-5 # lump sum tax

        # f. grids         
        par.a_max = 500.0 # maximum point in grid for a
        par.Na = 100 # number of grid points

        # h. shocks
        par.jump_G = 0.00 # initial jump
        par.rho_G = 0.00 # AR(1) coefficeint
        par.std_G = 0.00 # std.

        # g. indirect approach: targets for stationary equilibrium
        par.r_ss_target = 0.01
        par.w_ss_target = 1.0

        # h. misc.
        par.max_iter_ell = 200 # maximum number of iterations when solving for ell 
        par.max_iter_solve = 50_000 # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating household problem
        
        par.tol_ell = 1e-12 # tolerance when solving for ell 
        par.tol_solve = 1e-12 # tolerance when solving household problem
        par.tol_simulate = 1e-12 # tolerance when simulating household problem
        par.tol_broyden = 1e-10 # tolerance when solving eq. system




        
    def allocate(self):
        """ allocate model """

        par = self.par

        # a. grids
        # par.eta_grid = np.zeros(par.Nfix) # time-invariant productivity
        # par.iota_grid = np.zeros(par.Nfix) #  typespecific lump sum tax

        # b. solution
        self.allocate_GE() # should always be called here

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss


    def calc_u(self):

        par = self.par
        path = self.path

        U =np.sum([par.beta**t * np.sum(path.u[t]*path.D[t]/np.sum(path.D[t])) for t in range(par.T)])

        return U 
