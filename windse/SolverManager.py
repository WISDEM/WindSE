"""
The SolverManager contains all the different ways to solve problems generated
in windse
"""

import __main__
import os

### Get the name of program importing this package ###
main_file = os.path.basename(__main__.__file__)

### This checks if we are just doing documentation ###
if main_file != "sphinx-build":
    from dolfin import *
    from sys import platform

    ### Import the cumulative parameters ###
    from windse import windse_parameters

    ### Check if we need dolfin_adjoint ###
    if windse_parameters["general"].get("dolfin_adjoint", False):
        from dolfin_adjoint import *

    ### This import improves the plotter functionality on Mac ###
    if platform == 'darwin':
        import matplotlib
        matplotlib.use('TKAgg')
    import matplotlib.pyplot as plt

    ### Improve Solver parameters ###
    parameters["std_out_all_processes"] = False;
    parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -fno-math-errno -march=native'        
    parameters["form_compiler"]["optimize"]     = True
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters['form_compiler']['representation'] = 'uflacs'
    parameters['form_compiler']['quadrature_degree'] = 6

class GenericSolver(object):
    """
    A GenericSolver contains on the basic functions required by all solver objects.
    """
    def __init__(self,problem):
        self.params = windse_parameters
        self.problem  = problem
        self.u_next,self.p_next = self.problem.up_next.split(True)
        self.first_save = True


    def Plot(self):
        """
        This function plots the solution functions using matplotlib and saves the 
        output to output/.../plots/u.pdf and output/.../plots/p.pdf
        """

        ### Create the path names ###
        folder_string = self.params.folder+"/plots/"
        u_string = self.params.folder+"/plots/u.pdf"
        p_string = self.params.folder+"/plots/p.pdf"

        ### Check if folder exists ###
        if not os.path.exists(folder_string): os.makedirs(folder_string)

        ### Plot the x component of velocity ###
        plot(self.u_next[0],title="Velocity in the x Direction")
        plt.savefig(u_string)
        plt.figure()

        ### Plot the pressure ###
        plot(self.p_next,title="Pressure")
        plt.savefig(p_string)
        plt.show()

    def Save(self,val=0):
        """
        This function saves the mesh and boundary markers to output/.../solutions/
        """

        print("Saving Solutions")
        if self.first_save:
            self.u_file = self.params.Save(self.u_next,"velocity",subfolder="solutions/",val=val)
            self.p_file = self.params.Save(self.p_next,"pressure",subfolder="solutions/",val=val)
            self.first_save = False
        else:
            self.params.Save(self.u_next,"velocity",subfolder="solutions/",val=val,file=self.u_file)
            self.params.Save(self.p_next,"pressure",subfolder="solutions/",val=val,file=self.p_file)
        print("Solutions Saved")

    def ChangeWindAngle(self,theta):
        """
        This function recomputes all necessary components for a new wind direction

        Args: 
            theta (float): The new wind angle in radians
        """
        self.problem.ChangeWindAngle(theta)

class SteadySolver(GenericSolver):
    """
    This solver is for solving the steady state problem

    Args: 
        problem (:meth:`windse.ProblemManager.GenericProblem`): a windse problem object.
    """
    def __init__(self,problem):
        super(SteadySolver, self).__init__(problem)

    def Solve(self,iter_val=0):
        """
        This solves the problem setup by the problem object.
        """

        ### Save Files before solve ###
        if "mesh" in self.params.outputs:
            self.problem.dom.Save(val=iter_val)
        if "initial_guess" in self.params.outputs:
            self.problem.bd.SaveInitialGuess(val=iter_val)
        if "height" in self.params.outputs:
            self.problem.bd.SaveHeight()
        if "turbine_force" in self.params.outputs:
            self.problem.farm.SaveTurbineForce(val=iter_val)



        ### Add some helper functions to solver options ###
        solver_parameters = {"newton_solver":{
                             "linear_solver": "mumps", 
                             "maximum_iterations": 15,
                             "error_on_nonconvergence": False,
                             "relaxation_parameter": 1.0}}

        # set_log_level(LogLevel.PROGRESS)
        print("Solving")
        # print(dir(self.problem.bd.bcs))
        solve(self.problem.F == 0, self.problem.up_next, self.problem.bd.bcs, solver_parameters=solver_parameters)
        print("Solved")
        self.u_next,self.p_next = self.problem.up_next.split(True)

        ### Save solutions ###

        if "solution" in self.params.outputs:
            self.Save(val=iter_val)

class MultiAngleSolver(SteadySolver):
    """
    This solver will solve the problem using the steady state solver for every
    angle in angles.

    Args: 
        problem (:meth:`windse.ProblemManager.GenericProblem`): a windse problem object.
        angles (list): A list of wind inflow directions.
    """ 

    def __init__(self,problem,angles):
        super(MultiAngleSolver, self).__init__(problem)
        self.orignal_solve = super(MultiAngleSolver, self).Solve
        self.angles = angles

    def Solve(self):
        for i, theta in enumerate(self.angles):
            print("Computing for Wind Angle: "+repr(theta))
            if i > 0 or not near(theta,self.problem.dom.wind_direction):
                print("Adjusting for Angle "+repr(i+1)+" of "+repr(len(self.angles)))
                self.ChangeWindAngle(theta)
            self.orignal_solve(iter_val=theta)
            print("Finished Angle "+repr(i+1)+" of "+repr(len(self.angles)))