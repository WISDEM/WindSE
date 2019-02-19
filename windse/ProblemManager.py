"""
The ProblemManager contains all of the 
different classes of problems that windse can solve
"""

import __main__
import os

### Get the name of program importing this package ###
main_file = os.path.basename(__main__.__file__)

### This checks if we are just doing documentation ###
if main_file != "sphinx-build":
    from dolfin import *
    import numpy as np

    ### Import the cumulative parameters ###
    from windse import windse_parameters

    ### Check if we need dolfin_adjoint ###
    if windse_parameters["general"].get("dolfin_adjoint", False):
        from dolfin_adjoint import *

class GenericProblem(object):
    """
    A GenericProblem contains on the basic functions required by all problem objects.
    
    Args: 
        domain (:meth:`windse.DomainManager.GenericDomain`): a windse domain object.
        windfarm (:meth:`windse.WindFarmManager.GenericWindFarmm`): a windse windfarm object.
        function_space (:meth:`windse.FunctionSpaceManager.GenericFunctionSpace`): a windse function space object.
        boundary_conditions (:meth:`windse.BoundaryManager.GenericBoundary`): a windse boundary object.
    """
    def __init__(self,domain,windfarm,function_space,boundary_data):
        ### save a reference of option and create local version specifically of domain options ###
        self.params = windse_parameters
        self.dom  = domain
        self.farm = windfarm
        self.fs   = function_space 
        self.bd  = boundary_data

class StabilizedProblem(GenericProblem):
    """
    The StabilizedProblem setup everything required for solving Navier-Stokes with 
    a stabilization term

    Args: 
        domain (:meth:`windse.DomainManager.GenericDomain`): a windse domain object.
        windfarm (:meth:`windse.WindFarmManager.GenericWindFarmm`): a windse windfarm object.
        function_space (:meth:`windse.FunctionSpaceManager.GenericFunctionSpace`): a windse function space object.
        boundary_conditions (:meth:`windse.BoundaryManager.GenericBoundary`): a windse boundary object.
    """
    def __init__(self,domain,windfarm,function_space,boundary_conditions):
        super(StabilizedProblem, self).__init__(domain,windfarm,function_space,boundary_conditions)

        ### Create the turbine force ###
        print("Creating Turbine Force")
        zeroF = Function(self.fs.V0)
        tf0 = self.farm.ModTurbineForce(self.fs.V0,self.dom.mesh)
        tf = as_vector((tf0,zeroF,zeroF))
        self.tf=tf
        print("Turbine Force Created")

        ### These constants will be moved into the params file ###
        nu_T_mod=Constant(2)
        nu = Constant(0.00005)
        f = Constant((0.,0.,0.))
        mlDenom = 6

        ### Create the test/trial/functions ###
        self.up_next = Function(self.fs.W)
        u_next,p_next = split(self.up_next)
        v,q = TestFunctions(self.fs.W)

        ### Set the initial guess ###
        ### (this will become a separate function.)
        self.up_next.assign(self.bd.u0)

        ### Calculate the stresses and viscosities ###
        S = sqrt(2.*inner(0.5*(grad(u_next)+grad(u_next).T),0.5*(grad(u_next)+grad(u_next).T)))

        ### Create l_mix based on distance to the ground ###
        l_mix = Function(self.fs.Q)
        l_mix.vector()[:] = np.divide(self.bd.z_dist_Q.vector()[:],mlDenom)

        ### Calculate nu_T
        nu_T=l_mix**2.*S


        ### Create the functional ###
        self.F = inner(grad(u_next)*u_next, v)*dx + (nu+nu_T+nu_T_mod)*inner(grad(u_next), grad(v))*dx - inner(div(v),p_next)*dx - inner(div(u_next),q)*dx - inner(f,v)*dx + inner(tf*u_next[0]**2,v)*dx 

        ### Add in the Stabilizing term ###
        eps=Constant(0.01)
        stab = - eps*inner(grad(q), grad(p_next))*dx - eps*inner(grad(q), dot(grad(u_next), u_next))*dx 
        self.F += stab
