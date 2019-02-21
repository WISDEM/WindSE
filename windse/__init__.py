""" 
This is the init file for WindSE. It handle importing all the
submodules and initializing the parameters.
"""

from windse.ParameterManager import windse_parameters

def initialize(loc):
    """
    This function initializes all the submodules in WindSE.

    Args:
        loc (str): This string is the location of the .yaml parameters file.

    """

    windse_parameters.Load(loc)

    global BoxDomain, RectangleDomain, ImportedDomain
    from windse.DomainManager import BoxDomain, RectangleDomain, ImportedDomain

    global GridWindFarm, RandomWindFarm, ImportedWindFarm
    from windse.WindFarmManager import GridWindFarm, RandomWindFarm, ImportedWindFarm

    global LinearFunctionSpace, TaylorHoodFunctionSpace2D
    from windse.FunctionSpaceManager import LinearFunctionSpace, TaylorHoodFunctionSpace2D

    global PowerInflow, UniformInflow, UniformInflow2D
    from windse.BoundaryManager import PowerInflow, UniformInflow, UniformInflow2D

    global StabilizedProblem, TaylorHoodProblem2D
    from windse.ProblemManager import StabilizedProblem, TaylorHoodProblem2D

    global SteadySolver
    from windse.SolverManager import SteadySolver

    global CreateControl, CreateAxialBounds, PowerFunctional
    from windse.OptimizationManager import CreateControl, CreateAxialBounds, PowerFunctional