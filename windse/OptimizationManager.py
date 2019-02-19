"""
The OptimizationManager submodule contains all the required function for
optimizing via dolfin-adjoint. To use dolfin-adjoin set::

    general: 
        dolfin_adjoint: True

in the param.yaml file.

Todo:
    * Read through an update the docstrings for these functions.
    * Create specific optimization classes.
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
    

def CreateControl(m):
    """
    Creates the controls from a list of values

    Args:
        m (list): a list of values to optimize.
    """
    return [Control(mm) for mm in m]

def CreateAxialBounds(m):
    """
    Creates the optimization bounds for axial induction.

    Args:
        m (list): a list of controls
    """
    ub=[]
    lb=[]
    for i in range(len(m)):
        lb.append(Constant(0.))
        ub.append(Constant(0.75))
        
    bounds = [lb,ub]
    return bounds

def PowerFunctional(tf,u):
    """
    Creates the power functional that will be optimized

    Args:
        tf (dolfin.Function): Turbine Force function
        u (dolfin.Function): Velocity vector.
    """
    #how to handle rotation?
    # J=Functional(tf*u[0]**3*dx)
    J=assemble(-tf[0]*u[0]**3*dx)

    return J
