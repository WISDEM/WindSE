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

        ### Append the farm ###
        self.params.full_farm = self.farm

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
        self.tf = self.farm.ModTurbineForce(self.fs,self.dom.mesh)
        print("Turbine Force Created")

        ### add some helper items for dolfin_adjoint_helper.py ###
        self.params.ground_fx = self.dom.Ground
        self.params.full_hh = self.farm.HH

        ### These constants will be moved into the params file ###
        # nu_T_mod=Constant(.2)
        nu = Constant(0.00005)
        f = Constant((0.,0.,0.))
        mlDenom = 7

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

        xt = Constant(0.0)
        yt = Constant(0.0)
        zt = Constant(0.0)

        # Gg = Function(self.fs.V0)

        # Gg.vector()[:]+=1.0
        # # print()(Gg([xt,yt,zt])+42.42)*
        # class TestTR(UserExpression):
        #     def __init__(self,x, y **kwargs):
        #         self.x = x
        #         self.y = y
        #         super(TestTR,self).__init__(**kwargs)
        #     def eval(self, values, x):
        #         values[0] = dom.Ground(self.x,self.y)
        #     def value_shape(self):
        #         return ()
        # FuncTest = TestTR(mx,my,degree=2)FuncTest([xt,yt,zt])*


        # FunctTest = Expression("G(xm,ym)",G=dom.Ground,x=mx_1,y=my_1)
        # FunctTest = Expression("G(xm,ym)",G=dom.Ground,x=mx_1,y=my_1)


        ### Create the functional ###
        self.F = inner(grad(u_next)*u_next, v)*dx + (nu+nu_T)*inner(grad(u_next), grad(v))*dx - inner(div(v),p_next)*dx - inner(div(u_next),q)*dx - inner(f,v)*dx + inner(self.tf*(u_next[0]**2+u_next[1]**2),v)*dx 

        ### Add in the Stabilizing term ###
        eps=Constant(0.01)
        stab = - eps*inner(grad(q), grad(p_next))*dx - eps*inner(grad(q), dot(grad(u_next), u_next))*dx 
        self.F += stab

class TaylorHoodProblem2D(GenericProblem):
    """
    The TaylorHoodProblem2D sets up everything required for solving Navier-Stokes with 
    in 2D

    Args: 
        domain (:meth:`windse.DomainManager.GenericDomain`): a windse domain object.
        windfarm (:meth:`windse.WindFarmManager.GenericWindFarmm`): a windse windfarm object.
        function_space (:meth:`windse.FunctionSpaceManager.GenericFunctionSpace`): a windse function space object.
        boundary_conditions (:meth:`windse.BoundaryManager.GenericBoundary`): a windse boundary object.
    """
    def __init__(self,domain,windfarm,function_space,boundary_conditions):
        super(TaylorHoodProblem2D, self).__init__(domain,windfarm,function_space,boundary_conditions)

        ### Create the turbine force ###
        print("Creating Turbine Force")
        tf = self.farm.ModTurbineForce2D(self.fs,self.dom.mesh)
        self.tf = tf
        print("Turbine Force Created")

        ### These constants will be moved into the params file ###
        nu = Constant(.0005)
        f = Constant((0.,0.))
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
        l_mix = Constant(self.farm.HH[0]/mlDenom)

        ### Calculate nu_T
        nu_T=l_mix**2.*S

        ### Create the functional ###
        self.F = inner(grad(u_next)*u_next, v)*dx + (nu+nu_T)*inner(grad(u_next), grad(v))*dx - inner(div(v),p_next)*dx - inner(div(u_next),q)*dx - inner(f,v)*dx + inner(tf*(u_next[0]**2+u_next[1]**2),v)*dx 
