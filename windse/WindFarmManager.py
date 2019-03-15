"""
The windfarm manager contains everything required to set up a 
windfarm.
"""

import __main__
import os

### Get the name of program importing this package ###
main_file = os.path.basename(__main__.__file__)

### This checks if we are just doing documentation ###
if main_file != "sphinx-build":
    from dolfin import *
    import numpy as np
    from sys import platform
    import math

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

class GenericWindFarm(object):
    """
    A GenericProblem contains on the basic functions required by all problem objects.
    
    Args: 
        dom (:meth:`windse.DomainManager.GenericDomain`): a windse domain object.
    """
    def __init__(self, dom):
        ### save a reference of option and create local version specifically of domain options ###
        self.params = windse_parameters
        self.dom = dom

    def Plot(self,show=True):
        """
        This function plots the locations of each wind turbine and
        saves the output to output/.../plots/

        :Keyword Arguments:
            * **show** (*bool*): Default: True, Set False to suppress output but still save.
        """
        ### Create the path names ###
        folder_string = self.params.folder+"/plots/"
        file_string = self.params.folder+"/plots/wind_farm.pdf"

        ### Check if folder exists ###
        if not os.path.exists(folder_string): os.makedirs(folder_string)

        ### Create a list that outlines the extent of the farm ###
        ex_list_x = [self.ex_x[0],self.ex_x[1],self.ex_x[1],self.ex_x[0],self.ex_x[0]]
        ex_list_y = [self.ex_y[0],self.ex_y[0],self.ex_y[1],self.ex_y[1],self.ex_y[0]]

        ### Generate and Save Plot ###
        plt.plot(ex_list_x,ex_list_y,c="r")
        p=plt.scatter(self.x,self.y,c=self.z)
        plt.xlim(self.dom.x_range[0],self.dom.x_range[1])
        plt.ylim(self.dom.y_range[0],self.dom.y_range[1])
        clb = plt.colorbar(p)
        clb.ax.set_ylabel('Hub Height')

        plt.title("Location of the Turbines")
        plt.savefig(file_string)
        if show:
            plt.show()

    def SaveTurbineForce(self,filename="mesh",filetype="pvd"):
        """
        This function saves the turbine force to be viewed later.

        Todo:
            Needs to be implemented.
        """
        pass

    def GetLocations(self):
        """
        This function gets three lists containing the x, y, and z locations
        of each turbine.

        Returns:
            x, y, z (lists): turbine coordinates
        """
        return self.x, self.y, self.z

    def PrintLocations(self):
        """
        This function prints out the  locations of each turbine.
        """
        for i in range(self.numturbs):
            print("Turbine "+repr(i+1)+": "+repr([self.x[i],self.y[i],self.z[i]]))

    def CalculateExtents(self):
        """
        This functions takes into consideration the turbine locations, diameters, 
        and hub heights to create lists that describe the extent of the windfarm.
        These lists are append to the parameters object.
        """
        ### Locate the extreme turbines ### 
        x_min = np.argmin(self.x)
        x_max = np.argmax(self.x)
        y_min = np.argmin(self.y)
        y_max = np.argmax(self.y)
        z_min = np.argmin(self.z)
        z_max = np.argmax(self.z)

        ### Calculate the extent of the farm ###
        self.ex_x = [self.x[x_min]-self.RD[x_min]/2.0,self.x[x_max]+self.RD[x_max]/2.0]
        self.ex_y = [self.y[y_min]-self.RD[y_min]/2.0,self.y[y_max]+self.RD[y_max]/2.0]
        self.ex_z = [min(self.ground),self.z[z_max]+self.RD[z_max]/2.0]

        ### Update the options ###
        self.params["wind_farm"]["ex_x"] = self.ex_x
        self.params["wind_farm"]["ex_y"] = self.ex_y
        self.params["wind_farm"]["ex_z"] = self.ex_z

    def CreateConstants(self):
        """
        This functions converts lists of locations and axial inductions
        into dolfin.Constants. This is useful in optimization.
        """
        # self.mx = self.x
        # self.my = self.y
        # self.mz = self.z
        # self.ma = self.a
        self.mx = []
        self.my = []
        self.mz = []
        self.ma = []
        self.myaw = []
        for i in range(self.numturbs):
            self.mx.append(Constant(self.x[i]))
            self.my.append(Constant(self.y[i]))
            self.mz.append(Constant(self.z[i]))
            self.ma.append(Constant(self.a[i]))
            self.myaw.append(Constant(self.yaw[i]))

        for i in range(self.numturbs):
            self.mx[i].rename("x"+repr(i),"x"+repr(i))
            self.my[i].rename("y"+repr(i),"y"+repr(i))
            self.mz[i].rename("z"+repr(i),"z"+repr(i))


    def UpdateConstants(self):
        """
        This function updates the optimization constants
        """
        # self.mx = self.x
        # self.my = self.y
        # self.mz = self.z
        for i in range(self.numturbs):
            self.mx[i].assign(self.x[i])
            self.my[i].assign(self.y[i])
            self.mz[i].assign(self.z[i])
            self.ma[i].assign(self.a[i])
            self.myaw[i].assign(self.yaw[i])


    def CreateLists(self):
        """
        This function creates lists from single values. This is useful
        when the params.yaml file defines only one type of turbine.
        """
        self.HH = np.full(self.numturbs,self.HH)
        self.RD = np.full(self.numturbs,self.RD)
        self.W = np.full(self.numturbs,self.W)
        self.radius = np.full(self.numturbs,self.radius)
        self.yaw = np.full(self.numturbs,self.yaw)

    def RotateFarm(self,angle):
        """
        This function rotates the position of each turbine. It does not change 
        the yaw of the turbines. 

        Args:
            angle (float): the rotation angle in radians
        """

        center = [sum(self.dom.x_range)/2.0,sum(self.dom.y_range)/2.0,sum(self.dom.z_range)/2.0]
        for i in range(self.numturbs):
            x = [self.x[i],self.y[i],self.z[i]]
            self.x[i] = math.cos(angle)*(x[0]-center[0]) - math.sin(angle)*(x[1]-center[1])+center[0]
            self.y[i] = math.sin(angle)*(x[0]-center[0]) + math.cos(angle)*(x[1]-center[1])+center[1]
            self.z[i] = self.HH[i]+self.dom.ground(self.x[i],self.y[i])[0]
            # self.angle[i] -= rot
        self.CalculateExtents()
        self.UpdateConstants()

    def YawTurbine(self,x,x0,yaw):
        """
        This function yaws the turbines when creating the turbine force.

        Args:
            x (dolfin.SpacialCoordinate): the space variable, x
            x0 (list): the location of the turbine to be yawed
            yaw (float): the yaw value in radians
        """
        xrot = math.cos(yaw)*(x[0]-x0[0]) - math.sin(yaw)*(x[1]-x0[1])
        yrot = math.sin(yaw)*(x[0]-x0[0]) + math.cos(yaw)*(x[1]-x0[1])
        zrot = x[2]-x0[2]
        return [xrot,yrot,zrot]

    def YawTurbine2D(self,x,x0,yaw):
        """
        This function yaws the turbines when creating the turbine force.

        Args:
            x (dolfin.SpacialCoordinate): the space variable, x
            x0 (list): the location of the turbine to be yawed
            yaw (float): the yaw value in radians
        """
        xrot = math.cos(yaw)*(x[0]-x0[0]) - math.sin(yaw)*(x[1]-x0[1])
        yrot = math.sin(yaw)*(x[0]-x0[0]) + math.cos(yaw)*(x[1]-x0[1])
        return [xrot,yrot]

    def RotatedTurbineForce(self,V,mesh):
        """
        This function Creates the turbine force. It is deprecated. 
        Use :meth:`windse.WindFarmManager.GenericWindFarm.ModTurbineForce`
        instead.

        Args:
            V (dolfin.FunctionSpace): The function space the turbine force will use.
            mesh (dolfin.mesh): The mesh

        Returns:
            tf (dolfin.Function): the turbine force.
        """
        ### Set some values from options ###
        ### (this could be cleaned up and some will be moved to the yaml file) ###
        A = self.params["wind_farm"]["A"]
        alpha = self.params["wind_farm"]["alpha"]
        beta = self.params["wind_farm"]["beta"]
        WTGexp = 4.0
        thickness = self.RD/12.
        radius = self.RD/2.0

        x=SpatialCoordinate(mesh)
        WTGbase = Constant(("1.0","0.0","0.0"))

        tf = Function(V)

        for i in range(self.numturbs):
            ### Rotate (need to be compatible for 2D) ###
            xrot = cos(alpha)*self.mx[i] - sin(alpha)*self.my[i]
            yrot = sin(alpha)*self.mx[i] + cos(alpha)*self.my[i]
            zrot = self.mz[i]
            tf = tf + 0.75*0.5*4.*A*self.ma[i]/(1.-self.ma[i])/beta*exp(-(((x[0] - xrot)/thickness)**WTGexp + (((x[1] - yrot)**2 + (x[2] - zrot)**2)/radius**2)**WTGexp))*WTGbase

        return tf

    def ModTurbineForce(self,fs,mesh):
        """
        This function creates a turbine force by applying 
        a spacial kernel to each turbine. This kernel is 
        created from the turbines location, yaw, thickness, diameter,
        and force density. Currently, force density is limit to a scaled
        version of 

        .. math::

            r\\sin(r),

        where :math:`r` is the distance from the center of the turbine.

        Args:
            V (dolfin.FunctionSpace): The function space the turbine force will use.
            mesh (dolfin.mesh): The mesh

        Returns:
            tf (dolfin.Function): the turbine force.

        Todo:
            * Setup a way to get the force density from file
        """

        x=SpatialCoordinate(mesh)

        tf_x=Function(fs.V0)
        tf_y=Function(fs.V1)
        tf_z=Function(fs.V2)

        for i in range(self.numturbs):
            x0 = [self.mx[i],self.my[i],self.mz[i]]
            yaw = self.myaw[i]
            W = self.W[i]/2.0
            R = self.RD[i]/2.0 
            ma = self.ma[i]

            ### Rotate and Shift the Turbine ###
            xs = self.YawTurbine(x,x0,yaw)

            ### Create the function that represents the Thickness of the turbine ###
            T_norm = 1.902701539733748
            T = exp(-pow((xs[0]/W),10.0))/(T_norm*W)

            ### Create the function that represents the Disk of the turbine
            D_norm = 2.884512175878827
            D = exp(-pow((pow((xs[1]/R),2)+pow((xs[2]/R),2)),5.0))/(D_norm*R**2.0)

            ### Create the function that represents the force ###
            # F = 0.75*0.5*4.*A*self.ma[i]/(1.-self.ma[i])/beta
            r = sqrt(xs[1]**2.0+xs[2]**2)
            F = 4.*0.5*(pi*R**2.0)*ma/(1.-ma)*(r/R*sin(pi*r/R)+0.5) * 1/(.81831)

            ### Combine and add to the total ###
            tf_x = tf_x + F*T*D*cos(yaw)
            tf_y = tf_y + F*T*D*sin(yaw)

        ### Project Turbine Force to save on Assemble time ###
        print("Projecting Turbine Force")
        tf_x = project(tf_x,fs.V0,solver_type='mumps')
        tf_y = project(tf_y,fs.V1,solver_type='mumps')  
        print("Turbine Force Projected")

        ## Assign the components to the turbine force ###
        tf = Function(fs.V)
        fs.VelocityAssigner.assign(tf,[tf_x,tf_y,tf_z])

        ## Save Turbine Force
        self.params.Save(tf,"tf",subfolder="functions/")

        tf = as_vector((tf_x,tf_y,tf_z))

        return tf

    def ModTurbineForce2D(self,fs,mesh):
        """
        This function creates a turbine force by applying 
        a spacial kernel to each turbine. This kernel is 
        created from the turbines location, yaw, thickness, diameter,
        and force density. Currently, force density is limit to a scaled
        version of 

        .. math::

            r\\sin(r),

        where :math:`r` is the distance from the center of the turbine.

        Args:
            V (dolfin.FunctionSpace): The function space the turbine force will use.
            mesh (dolfin.mesh): The mesh

        Returns:
            tf (dolfin.Function): the turbine force.

        Todo:
            * Setup a way to get the force density from file
        """

        x=SpatialCoordinate(mesh)

        tf_x=Function(fs.V0)
        tf_y=Function(fs.V1)

        for i in range(self.numturbs):
            x0 = [self.mx[i],self.my[i]]
            yaw = self.myaw[i]
            W = self.W[i]/2.0
            R = self.RD[i]/2.0 
            ma = self.ma[i]

            ### Rotate and Shift the Turbine ###
            xs = self.YawTurbine2D(x,x0,yaw)

            ### Create the function that represents the Thickness of the turbine ###
            T_norm = 1.902701539733748
            T = exp(-pow((xs[0]/W),10.0))/(T_norm*W)

            ### Create the function that represents the Disk of the turbine
            D_norm = 2.884512175878827
            D = exp(-pow((pow((xs[1]/R),2)),5.0))/(D_norm*R**2.0)

            ### Create the function that represents the force ###
            # F = 0.75*0.5*4.*A*self.ma[i]/(1.-self.ma[i])/beta
            r = sqrt(xs[1]**2.0)
            # F = 4.*0.5*(pi*R**2.0)*ma/(1.-ma)*(r/R*sin(pi*r/R)+0.5)
            F = 4.*0.5*(pi*R**2.0)*ma/(1.-ma)*(r/R*sin(pi*r/R)+0.5)

            ### Combine and add to the total ###
            tf_x = tf_x + F*T*D*cos(yaw)
            tf_y = tf_y + F*T*D*sin(yaw)

        ### Project Turbine Force to save on Assemble time ###
        print("Projecting Turbine Force")
        tf_x = project(tf_x,fs.V0,solver_type='mumps')
        tf_y = project(tf_y,fs.V1,solver_type='mumps')        
        print("Turbine Force Projected")

        ### Assign the components to the turbine force ###
        tf = Function(fs.V)
        fs.VelocityAssigner.assign(tf,[tf_x,tf_y])

        ### Save Turbine Force
        self.params.Save(tf,"tf",subfolder="functions/")

        return tf


    def ModTurbineForceTest(self,V,mesh):
        """
        This is an experimental version of :meth:`windse.WindFarmManager.GenericWindFarm.ModTurbineForce`.
        The intention is to build the turbine force using only
        numpy operations. This should be faster than projecting. 

        Args:
            V (dolfin.FunctionSpace): The function space the turbine force will use.
            mesh (dolfin.mesh): The mesh

        Returns:
            tf (dolfin.Function): the turbine force.
        """
        x = V.tabulate_dof_coordinates().T

        tf=Function(V)

        for i in range(self.numturbs):
            x0 = [self.mx[i],self.my[i],self.mz[i]]
            yaw = self.myaw[i]
            W = self.W[i]/2.0
            R = self.RD[i]/2.0 
            ma = self.ma[i]

            ### Rotate and Shift the Turbine ###
            xs = self.YawTurbine(x,x0,yaw)
            print(x0)
            ### Create the function that represents the Thickness of the turbine ###
            T_norm = 1.902701539733748
            T = np.exp(-(xs[0]/W)**10.0)/(T_norm*W)

            ### Create the function that represents the Disk of the turbine
            D_norm = 2.884512175878827
            D = np.exp(-((xs[1]/R)**2+(xs[2]/R)**2)**5.0)/(D_norm*R**2.0)

            ### Create the function that represents the force ###
            # F = 0.75*0.5*4.*A*self.ma[i]/(1.-self.ma[i])/beta
            r = np.sqrt(xs[1]**2.0+xs[2]**2)
            F = 0.75*4.*0.5*(pi*R**2.0)*ma/(1.-ma)*(r/R*np.sin(pi*r/R)+0.5)

            ### Combine and add to the total ###
            tf.vector()[:] += F*T*D
            # tf = tf + F*T*D

        ### Save Lmix because its interesting
        self.params.Save(tf,"functions/tf")

        return tf

class GridWindFarm(GenericWindFarm):
    """
    A GridWindFarm produces turbines on a grid. The params.yaml file determines
    how this grid is set up.

    Example:
        In the .yaml file you need to define::

            wind_farm: 
                #                     # Description              | Units
                HH: 90                # Hub Height               | m
                RD: 126.0             # Turbine Diameter         | m
                thickness: 10.5       # Effective Thickness      | m
                yaw: 0.0              # Yaw                      | rads
                axial: 0.33           # Axial Induction          | -
                ex_x: [-1500, 1500]   # x-extent of the farm     | m
                ex_y: [-1500, 1500]   # y-extent of the farm     | m
                grid_rows: 6          # Number of rows           | -
                grid_cols: 6          # Number of columns        | -

        This will produce a 6x6 grid of turbines equally spaced within the 
        region [-1500, 1500]x[-1500, 1500].

    Args: 
        dom (:meth:`windse.DomainManager.GenericDomain`): a windse domain object.
    """
    def __init__(self,dom):
        super(GridWindFarm, self).__init__(dom)

        ### Initialize Values from Options ###
        self.grid_rows = self.params["wind_farm"]["grid_rows"]
        self.grid_cols = self.params["wind_farm"]["grid_cols"]
        self.numturbs = self.grid_rows * self.grid_cols
        self.params["wind_farm"]["numturbs"] = self.numturbs

        self.HH = [self.params["wind_farm"]["HH"]]*self.numturbs
        self.RD = [self.params["wind_farm"]["RD"]]*self.numturbs
        self.W = [self.params["wind_farm"]["thickness"]]*self.numturbs
        self.yaw = [self.params["wind_farm"]["yaw"]]*self.numturbs
        self.axial = [self.params["wind_farm"]["axial"]]*self.numturbs
        self.radius = self.RD[0]/2.0

        self.ex_x = self.params["wind_farm"]["ex_x"]
        self.ex_y = self.params["wind_farm"]["ex_y"]
        
        ### Create the x and y coords ###
        self.grid_x = np.linspace(self.ex_x[0]+self.radius,self.ex_x[1]-self.radius,self.grid_cols)
        self.grid_y = np.linspace(self.ex_y[0]+self.radius,self.ex_y[1]-self.radius,self.grid_rows)

        ### Use the x and y coords to make a mesh grid ###
        self.x, self.y = np.meshgrid(self.grid_x,self.grid_y)
        self.x = self.x.flatten()
        self.y = self.y.flatten()

        ### Calculate Ground Heights ###
        self.ground = dom.Ground(self.x,self.y)

        ### Create the z coords ###
        self.z = np.full(self.numturbs,self.HH)
        self.z = np.add(self.z, self.ground)

        ### Update the extent in the z direction ###
        self.ex_z = [min(self.ground),max(self.z)+self.RD]
        self.params["wind_farm"]["ex_z"] = self.ex_z

        ### Axial induction Factor ###
        self.a = np.full(self.numturbs,self.axial)

        ### Convert the lists into lists of dolfin Constants ###
        ### This value should be added into the options ###
        self.CreateConstants() 

        ### Convert the constant parameters to lists ###
        self.CreateLists()

class RandomWindFarm(GenericWindFarm):
    """
    A RandomWindFarm produces turbines located randomly with a defined 
    range. The params.yaml file determines how this grid is set up.

    Example:
        In the .yaml file you need to define::

            wind_farm: 
                #                     # Description              | Units
                HH: 90                # Hub Height               | m
                RD: 126.0             # Turbine Diameter         | m
                thickness: 10.5       # Effective Thickness      | m
                yaw: 0.0              # Yaw                      | rads
                axial: 0.33           # Axial Induction          | -
                ex_x: [-1500, 1500]   # x-extent of the farm     | m
                ex_y: [-1500, 1500]   # y-extent of the farm     | m
                numturbs: 36          # Number of Turbines       | -
                seed: 15              # Random Seed for Numpy    | -

        This will produce a 36 turbines randomly located within the 
        region [-1500, 1500]x[-1500, 1500]. The seed is optional but 
        useful for reproducing test.

    Args: 
        dom (:meth:`windse.DomainManager.GenericDomain`): a windse domain object.
    """
    def __init__(self,dom):
        super(RandomWindFarm, self).__init__(dom)

        ### Initialize Values from Options ###
        self.numturbs = self.params["wind_farm"]["numturbs"]
        self.HH = self.params["wind_farm"]["HH"]
        self.RD = self.params["wind_farm"]["RD"]
        self.W = self.params["wind_farm"]["thickness"]
        self.yaw = self.params["wind_farm"]["yaw"]
        self.axial = self.params["wind_farm"]["axial"]
        # self.radius = self.RD/2.0

        self.ex_x = self.params["wind_farm"]["ex_x"]
        self.ex_y = self.params["wind_farm"]["ex_y"]

        self.seed = self.params["wind_farm"].get("seed",None)

        ### Check if random seed is set ###
        if self.seed is not None:
            np.random.seed(self.seed)

        ### Create the x and y coords ###
        self.x = np.random.uniform(self.ex_x[0]+self.radius,self.ex_x[1]-self.radius,self.numturbs)
        self.y = np.random.uniform(self.ex_y[0]+self.radius,self.ex_y[1]-self.radius,self.numturbs)
        
        ### Calculate Ground Heights ###
        self.ground = dom.Ground(self.x,self.y)

        ### Create the z coords ###
        self.z = np.full(self.numturbs,self.HH)
        self.z = np.add(self.z, self.ground)

        ### Axial induction Factor ###
        self.a = np.full(self.numturbs,self.axial)

        ### Convert the lists into lists of dolfin Constants ###
        self.CreateConstants() 

        ### Convert the constant parameters to lists ###
        self.CreateLists()

class ImportedWindFarm(GenericWindFarm):
    """
    A ImportedWindFarm produces turbines located based on a text file.
    The params.yaml file determines how this grid is set up.

    Example:
        In the .yaml file you need to define::

            wind_farm: 
                imported: true
                path: "inputs/wind_farm.txt"

        The "wind_farm.txt" needs to be set up like this::

            #    x      y     HH           Yaw Diameter Thickness Axial_Induction
            200.00 0.0000 80.000  0.0000000000      126      10.5            0.33
            800.00 0.0000 80.000  0.0000000000      126      10.5            0.33

        The first row isn't necessary. Each row defines a different turbine.

    Args: 
        dom (:meth:`windse.DomainManager.GenericDomain`): a windse domain object.
    """
    def __init__(self,dom):
        super(ImportedWindFarm, self).__init__(dom)
        
        ### Import the data from path ###
        print("Loading Wind Farm Data")
        self.path = self.params["wind_farm"]["path"]
        raw_data = np.loadtxt(self.path,comments="#")
        print("Wind Farm Loaded")

        ### Parse the data ###
        self.x     = raw_data[:,0] 
        self.y     = raw_data[:,1]
        self.HH    = raw_data[:,2]
        self.yaw   = raw_data[:,3]
        self.RD    = raw_data[:,4]
        self.W     = raw_data[:,5]
        self.a     = raw_data[:,6]

        ### Update the options ###
        self.numturbs = len(self.x)
        self.params["wind_farm"]["numturbs"] = self.numturbs

        ### Calculate Ground Heights ###
        print("Calculating Hub Heights")
        self.ground   = dom.Ground(self.x,self.y)
        self.z        = np.add(self.HH,self.ground)
        print("Hub Heights Calculated")

        ### Calculate the extent of the farm ###
        self.CalculateExtents()

        ### Convert the lists into lists of dolfin Constants ###
        self.CreateConstants() 