""" 
The BoundaryManager submodule contains the classes required for 
defining the boundary conditions. The boundaries need to be numbered
as follows:

    * 1: Top
    * 2: Bottom
    * 3: Front
    * 4: Back
    * 5: Left
    * 6: Right

Currently, the inflow direction is the Left boundary.
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

class PowerInflow(object):
    """
    PowerInflow creates a set of boundary conditions where the x-component
    of velocity follows a power law. Currently the function is 

    .. math::

        u_x=16.0 \\left( \\frac{z-z_0}{z_1-z_0} \\right)^{1/4}.
        
    where :math:`z_0` is the ground and :math:`z_1` is the top of the domain.

    Args:
        dom (:class:`windse.DomainManager.GenericDomain`): A windse domain object.
        fs (:class:`windse.FunctionSpaceManager.GenericFunctionSpace`): 
            A windse function space object

    Todo:
        * Make the max velocity an input
        * Make the power an input
    """
    def __init__(self,dom,fs):
        self.params = windse_parameters

        ### Calculate the distance to the ground for the Q function space ###
        self.z_dist_Q = Function(fs.Q)
        Q_coords = fs.Q.tabulate_dof_coordinates()
        z_dist_Q_val = np.zeros(len(Q_coords))
        for i in range(len(Q_coords)):
            z_dist_Q_val[i] = Q_coords[i,2]-dom.Ground(Q_coords[i,0],Q_coords[i,1])
        self.z_dist_Q.vector()[:]=z_dist_Q_val
        scaled_z_dist_Q_val = np.abs(np.divide(self.z_dist_Q.vector()[:],(dom.z_range[1]-dom.z_range[0])))

        ### Save distance to ground
        self.params.Save(self.z_dist_Q,"height",subfolder="functions/")


        ### Calculate the distance to the ground for the Q function space ###
        self.z_dist_V = Function(fs.V)
        V_coords = fs.V.tabulate_dof_coordinates()
        z_dist_V_val = np.zeros(len(V_coords))
        for i in range(len(V_coords)):
            z_dist_V_val[i] = V_coords[i,2]-dom.Ground(V_coords[i,0],V_coords[i,1])
        self.z_dist_V.vector()[:]=z_dist_V_val
        z_dist_V0,z_dist_V1,z_dist_V2 = self.z_dist_V.split(True)

        ### Create the Velocity Function ###
        ux = Function(fs.V0)
        uy = Function(fs.V1)
        uz = Function(fs.V2)
        scaled_z_dist_val = np.abs(np.divide(z_dist_V0.vector()[:],(dom.z_range[1]-dom.z_range[0])))
        ux.vector()[:] = np.multiply(16.0,np.power(scaled_z_dist_Q_val,1./4.))
        # ux.vector()[:] = np.multiply(16.0,np.power(scaled_z_dist_val,1./4.))

        ### Assigning Velocity
        self.bc_velocity = Function(fs.V)
        fs.VelocityAssigner.assign(self.bc_velocity,[ux,uy,uz])

        ### Save Velocity because its interesting
        self.params.Save(self.bc_velocity,"initialCondition",subfolder="functions/")

        ### Create Pressure Boundary Function
        self.bc_pressure = Function(fs.Q)

        ### Create Initial Guess
        self.u0 = Function(fs.W)
        fs.SolutionAssigner.assign(self.u0,[self.bc_velocity,self.bc_pressure])

        ### Create the equations need for defining the boundary conditions ###
        ### this is sloppy and will be cleaned up.
        ### Inflow is always from the front
        print("Creating Boundary Conditions")
        bcTop    = self.bc_velocity
        bcBottom = Constant((0.0,0.0,0.0))
        bcFront  = self.bc_velocity
        bcBack   = None
        bcLeft   = self.bc_velocity
        bcRight  = self.bc_velocity

        self.bcs_eqns = [bcTop,bcBottom,bcFront,bcBack,bcLeft,bcRight] 

        ### Create the boundary conditions ###
        self.bcs = []
        for i in range(len(self.bcs_eqns)):
            if self.bcs_eqns[i] is not None:
                self.bcs.append(DirichletBC(fs.W.sub(0), self.bcs_eqns[i], dom.boundary_markers, i+1))
        print("Boundary Conditions Created")

# class LogLayerInflow(object):
#     def __init__(self,dom,fs):