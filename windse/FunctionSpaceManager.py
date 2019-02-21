"""
The FunctionSpaceManager contains all the different types of function
spaces required for solve multiple classes of problems.
"""

import __main__
import os

### Get the name of program importing this package ###
main_file = os.path.basename(__main__.__file__)

### This checks if we are just doing documentation ###
if main_file != "sphinx-build":
    from dolfin import *

    ### Import the cumulative parameters ###
    from windse import windse_parameters

    ### Check if we need dolfin_adjoint ###
    if windse_parameters["general"].get("dolfin_adjoint", False):
        from dolfin_adjoint import *

class LinearFunctionSpace(object):
    """
    The LinearFunctionSpace is made up of a vector function space for velocity
    and a scaler space for pressure. Both spaces are "CG1" or Linear Lagrange elements.
    """
    def __init__(self,dom):

        ### Create the function space ###
        print("Creating Function Spaces")
        V = VectorElement('Lagrange', dom.mesh.ufl_cell(), 1) 
        Q = FiniteElement('Lagrange', dom.mesh.ufl_cell(), 1)
        self.W = FunctionSpace(dom.mesh, MixedElement([V,Q]))
        self.V = self.W.sub(0).collapse()
        self.Q = self.W.sub(1).collapse()
        self.V0 = self.V.sub(0).collapse() 
        self.V1 = self.V.sub(1).collapse() 
        self.V2 = self.V.sub(2).collapse()

        self.VelocityAssigner = FunctionAssigner(self.V,[self.V0,self.V1,self.V2])
        self.SolutionAssigner = FunctionAssigner(self.W,[self.V,self.Q])
        # exit()
        print("Function Spaces Created")

class TaylorHoodFunctionSpace2D(object):
    """
    The TaylorHoodFunctionSpace2D is made up of a vector function space for velocity
    and a scalar space for pressure. The velocity function space is piecewise quadratic
    and the pressure function space is piecewise linear.
    """
    def __init__(self,dom):

        ### Create the function space ###
        print("Creating Function Spaces")
        V = VectorElement('Lagrange', dom.mesh.ufl_cell(), 2) 
        Q = FiniteElement('Lagrange', dom.mesh.ufl_cell(), 1)
        self.W = FunctionSpace(dom.mesh, MixedElement([V,Q]))
        self.V = self.W.sub(0).collapse()
        self.Q = self.W.sub(1).collapse()
        self.V0 = self.V.sub(0).collapse() 
        self.V1 = self.V.sub(1).collapse() 

        self.VelocityAssigner = FunctionAssigner(self.V,[self.V0,self.V1])
        self.SolutionAssigner = FunctionAssigner(self.W,[self.V,self.Q])
        # exit()
        print("Function Spaces Created")

