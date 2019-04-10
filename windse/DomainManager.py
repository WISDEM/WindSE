"""
The DomainManager submodule contains the various classes used for 
creating different types of domains
"""

import __main__
import os

### Get the name of program importing this package ###
main_file = os.path.basename(__main__.__file__)

### This checks if we are just doing documentation ###
if main_file != "sphinx-build":
    from dolfin import *
    import copy
    import warnings
    import os
    from sys import platform
    import numpy as np
    from scipy.interpolate import interp2d, interp1d,RectBivariateSpline

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

    ### This parameter allows for refining the mesh functions ###
    parameters["refinement_algorithm"] = "plaza_with_parent_facets"


class GenericDomain(object):
    """
    A GenericDomain contains on the basic functions required by all domain objects
    """

    def __init__(self):
        ### save a reference of option and create local version specifically of domain options ###
        self.params = windse_parameters

    def Plot(self):
        """
        This function plots the domain using matplotlib and saves the 
        output to output/.../plots/mesh.pdf
        """

        ### Create the path names ###
        folder_string = self.params.folder+"/plots/"
        file_string = self.params.folder+"/plots/mesh.pdf"

        ### Check if folder exists ###
        if not os.path.exists(folder_string): os.makedirs(folder_string)

        p=plot(self.mesh)
        plt.savefig(file_string)
        plt.show()

    def Save(self,filename="domain",filetype="pvd"):
        """
        This function saves the mesh and boundary markers to output/.../mesh/

        :Keyword Arguments:
            * **filename** (*str*): the file name that preappends the meshes
            * **filetype** (*str*): the file type to save (pvd or xml)
        """
        folder_string = self.params.folder+"/mesh/"
        mesh_string = self.params.folder+"/mesh/"+filename+"_mesh."+filetype
        bc_string = self.params.folder+"/mesh/"+filename+"_boundaries."+filetype

        ### Check if folder exists ###
        if not os.path.exists(folder_string): os.makedirs(folder_string)

        print("Saving Mesh")
        ### Save Mesh ###
        file = File(mesh_string)
        file << self.mesh

        ### Save Boundary Function ###
        file = File(bc_string)
        file << self.boundary_markers
        print("Mesh Saved")

    def Refine(self,num,region=None):
        """
        This function can be used to refine the mesh. If a region is
        specified, the refinement is local

        Args:
            num (int): the number of times to refine

        :Keyword Arguments:
            * **region** (*list*): the specific region to refine in the form: [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
        """

        for i in range(num):
            if region is not None:
                cell_f = MeshFunction('bool', self.mesh, self.mesh.geometry().dim(),False)
                for cell in cells(self.mesh):
                    if between(cell.midpoint()[0],tuple(region[0])) and \
                       between(cell.midpoint()[1],tuple(region[1])) and \
                       between(cell.midpoint()[2],tuple(region[2])):
                        cell_f[cell] = True
                self.mesh = refine(self.mesh,cell_f)
                self.boundary_markers = adapt(self.boundary_markers,self.mesh)
            else:
                self.mesh = refine(self.mesh)
                self.boundary_markers = adapt(self.boundary_markers,self.mesh)

    def Warp(self,h,s):
        """
        This function warps the mesh to shift more cells towards the ground. 
        is achieved by spliting the domain in two and moving the cells so 
        that a percentage of them are below the split.

        Args:
            h (float): the height that split occurs
            s (float); the percent below split in the range [0,1)
        """

        z = copy.deepcopy(self.mesh.coordinates()[:,2])
        z0 = self.z_range[0]
        z1 = self.z_range[1]
        # cubic_spline = interp1d([z0,a-r,a+r,z1],[z0,a-(1-s)*r,a+(1-s)*r,z1])
        cubic_spline = interp1d([z0,h+s*(z1-(h)),z1],[z0,h,z1])
        # x = np.linspace(z0,z1,100)
        # y = cubic_spline(x)
        # plt.plot(x,y)
        # plt.show()
        # exit()

        z = cubic_spline(z)
        self.mesh.coordinates()[:,2]=z

    def WarpNonlinear(self,s):
        """
        This function warps the mesh to shift more cells towards the ground. 
        The cells are shifted based on the function:

        .. math::

            z_new = z_0 + (z_1-z_0) \\left( \\frac{z_old-z_0}{z_1-z_0} \\right)^{s}.

        where :math:`z_0` is the ground and :math:`z_1` is the top of the domain.

        Args:
            s (float): compression strength
        """
        z=self.mesh.coordinates()[:,2].copy()
        z0 = self.z_range[0]
        z1 = self.z_range[1]
        z1 = z0 + (z1 - z0)*((z-z0)/(z1-z0))**s
        self.mesh.coordinates()[:,2]=z1

class BoxDomain(GenericDomain):
    """
    A box domain is simply a 3D rectangular prism. This box is defined
    by 6 parameters in the param.yaml file. 

    Example:
        In the yaml file define::

            domain: 
                #                      # Description           | Units
                x_range: [-2500, 2500] # x-range of the domain | m
                y_range: [-2500, 2500] # y-range of the domain | m
                z_range: [0.04, 630]   # z-range of the domain | m
                nx: 10                 # Number of x-nodes     | -
                ny: 10                 # Number of y-nodes     | -
                nz: 2                  # Number of z-nodes     | -

        This will produce a box with corner points (-2500,-2500,0.04) 
        to (2500,2500,630). The mesh will have *nx* nodes in the *x*-direction,
        *ny* in the *y*-direction, and *nz* in the *z*-direction.
    """

    def __init__(self):
        super(BoxDomain, self).__init__()

        ### Initialize values from Options ###
        self.x_range = self.params["domain"]["x_range"]
        self.y_range = self.params["domain"]["y_range"]
        self.z_range = self.params["domain"]["z_range"]
        self.nx = self.params["domain"]["nx"]
        self.ny = self.params["domain"]["ny"]
        self.nz = self.params["domain"]["nz"]

        ### Create mesh ###
        start = Point(self.x_range[0], self.y_range[0], self.z_range[0])
        stop  = Point(self.x_range[1], self.y_range[1], self.z_range[1])
        self.mesh = BoxMesh(start, stop, self.nx, self.ny, self.nz)

        ### Define Boundary Subdomains ###
        top     = CompiledSubDomain("near(x[2], z1) && on_boundary",z1 = self.z_range[1])
        bottom  = CompiledSubDomain("near(x[2], z0) && on_boundary",z0 = self.z_range[0])
        front   = CompiledSubDomain("near(x[0], x0) && on_boundary",x0 = self.x_range[0])
        back    = CompiledSubDomain("near(x[0], x1) && on_boundary",x1 = self.x_range[1])
        left    = CompiledSubDomain("near(x[1], y0) && on_boundary",y0 = self.y_range[0])
        right   = CompiledSubDomain("near(x[1], y1) && on_boundary",y1 = self.y_range[1])
        self.boundary_subdomains = [top,bottom,front,back,left,right]

        ### Generate the boundary markers for boundary conditions ###
        self.boundary_markers = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.boundary_markers.set_all(0)
        for i in range(len(self.boundary_subdomains)):
            self.boundary_subdomains[i].mark(self.boundary_markers, i+1)

    def Ground(self,x,y):
        """
        Ground returns the ground height given an (*x*, *y*) coordinate.

        Args:
            x (float/list): *x* location within the domain
            y (float/list): *y* location within the domain

        Returns:
            float/list: corresponding z coordinates of the ground.

        """

        if (isinstance(x,list) and isinstance(y,list)) or (isinstance(x,np.ndarray) and isinstance(y,np.ndarray)):
            nx = len(x)
            ny = len(y)
            if nx != ny:
                raise ValueError("Length mismatch: len(x)="+repr(nx)+", len(y)="+repr(ny))
            else:
                return np.full(nx,self.z_range[0])
        else:
            return self.z_range[0]

class RectangleDomain(GenericDomain):
    """
    A rectangle domain is simply a 2D rectangle. This mesh is defined
    by 4 parameters in the param.yaml file. 

    Example:
        In the yaml file define::

            domain: 
                #                      # Description           | Units
                x_range: [-2500, 2500] # x-range of the domain | m
                y_range: [-2500, 2500] # y-range of the domain | m
                nx: 10                 # Number of x-nodes     | -
                ny: 10                 # Number of y-nodes     | -

        This will produce a rectangle with corner points (-2500,-2500) 
        to (2500,2500). The mesh will have *nx* nodes in the *x*-direction,
        and *ny* in the *y*-direction.

    Todo:
        Properly implement a RectangleDomain and 2D in general.
    """

    def __init__(self):
        super(RectangleDomain, self).__init__()

        ### Initialize values from Options ###
        self.x_range = self.params["domain"]["x_range"]
        self.y_range = self.params["domain"]["y_range"]
        self.nx = self.params["domain"]["nx"]
        self.ny = self.params["domain"]["ny"]

        ### Create mesh ###
        start = Point(self.x_range[0], self.y_range[0])
        stop  = Point(self.x_range[1], self.y_range[1])
        self.mesh = RectangleMesh(start, stop, self.nx, self.ny)

        ### Define Boundary Subdomains ###
        front   = CompiledSubDomain("near(x[0], x0) && on_boundary",x0 = self.x_range[0])
        back    = CompiledSubDomain("near(x[0], x1) && on_boundary",x1 = self.x_range[1])
        left    = CompiledSubDomain("near(x[1], y0) && on_boundary",y0 = self.y_range[0])
        right   = CompiledSubDomain("near(x[1], y1) && on_boundary",y1 = self.y_range[1])
        self.boundary_subdomains = [front,back,left,right]

        ### Generate the boundary markers for boundary conditions ###
        self.boundary_markers = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.boundary_markers.set_all(0)
        for i in range(len(self.boundary_subdomains)):
            self.boundary_subdomains[i].mark(self.boundary_markers, i+1)

    def Ground(self,x,y):
        if (isinstance(x,list) and isinstance(y,list)) or (isinstance(x,np.ndarray) and isinstance(y,np.ndarray)):
            nx = len(x)
            ny = len(y)
            if nx != ny:
                raise ValueError("Length mismatch: len(x)="+repr(nx)+", len(y)="+repr(ny))
            else:
                return np.full(nx,0)

class ImportedDomain(GenericDomain):
    """
    This class generates a domain from imported files. This mesh is defined
    by 2 parameters in the param.yaml file. 

    Example:
        In the yaml file define::

            domain: 
                path: "Mesh_data/"
                filetype: "xml.gz"

        The supported filetypes are "xml.gz" and "h5". For "xml.gz" 3 files are
        required: 

            * mesh.xml.gz - this contains the mesh in a format dolfin can handle
            * boundaries.xml.gz - this contains the facet markers that define where the boundaries are
            * topology.txt - this contains the data for the ground topology. 
                It assumes that the coordinates are from a uniform mesh.
                It contains three column: x, y, z. The x and y columns contain 
                just the unique values. The z column contains the ground values
                for every combination of x and y. The first row must be the number
                of points in the x and y direction. Here is an example for z=x+y/10::

                    3 3 9
                    0 0 0.0
                    1 1 0.1
                    2 2 0.2
                        1.0
                        1.1
                        1.2
                        2.0
                        2.1
                        2.2

    """

    def __init__(self):
        super(ImportedDomain, self).__init__()

        ### Get the file type for the mesh (h5, xml.gz) ###
        self.filetype = self.params["domain"].get("filetype", "xml.gz")

        ### Import data from Options ###
        if "path" in self.params["domain"]:
            self.path = self.params["domain"]["path"]
            self.mesh_path  = self.path + "mesh." + self.filetype
            if self.filetype == "xml.gz":
                self.boundary_path = self.path + "boundaries." + self.filetype
            self.typo_path  = self.path + "topology.txt"
        else:
            self.mesh_path = self.params["domain"]["mesh_path"]
            if self.filetype == "xml.gz":
                self.boundary_path = self.params["domain"]["boundary_path"]
            self.typo_path  = self.params["domain"]["typo_path"]

        ### Create the mesh ###
        print("Importing Mesh")
        if self.filetype == "h5":
            self.mesh = Mesh()
            hdf5 = HDF5File(self.mesh.mpi_comm(), self.mesh_path, 'r')
            hdf5.read(self.mesh, '/mesh', False)
        elif self.filetype == "xml.gz":
            self.mesh = Mesh(self.mesh_path)
        else:
            raise ValueError("Supported mesh types: h5, xml.gz.")
        print("Mesh Imported")

        ### Calculate the range of the domain and push to options ###
        self.x_range = [min(self.mesh.coordinates()[:,0]),max(self.mesh.coordinates()[:,0])]
        self.y_range = [min(self.mesh.coordinates()[:,1]),max(self.mesh.coordinates()[:,1])]
        self.z_range = [min(self.mesh.coordinates()[:,2]),max(self.mesh.coordinates()[:,2])]
        self.params["domain"]["x_range"] = self.x_range
        self.params["domain"]["y_range"] = self.y_range
        self.params["domain"]["z_range"] = self.z_range

        ### Load the boundary markers ###
        print("Importing Boundary Markers")
        if self.filetype == "h5":
            self.boundary_markers = MeshFunction("size_t", self.mesh, self.mesh.geometry().dim()-1)
            hdf5.read(self.boundary_markers, "/boundaries")
        elif self.filetype == "xml.gz":
            self.boundary_markers = MeshFunction("size_t", self.mesh, self.boundary_path)
        print("Markers Imported")

        ### Create the interpolation function for the ground ###
        self.topography = np.loadtxt(self.typo_path)
        x_data = self.topography[1:,0]
        y_data = self.topography[1:,1]
        z_data = self.topography[1:,2]
        
        x_data = np.sort(np.unique(x_data))
        y_data = np.sort(np.unique(y_data))
        z_data = np.reshape(z_data,(int(self.topography[0,0]),int(self.topography[0,1])))

        print("Creating Interpolating Function")
        self.topography_interpolated = RectBivariateSpline(x_data,y_data,z_data.T)
        print("Interpolating Function Created")

    def Ground(self,x,y):
        """
        Ground returns the ground height given an (*x*, *y*) coordinate.

        Args:
            x (float/list): *x* location within the domain
            y (float/list): *y* location within the domain

        Returns:
            float/list: corresponding z coordinates of the ground.

        """
        if (isinstance(x,list) and isinstance(y,list)) or (isinstance(x,np.ndarray) and isinstance(y,np.ndarray)):
            nx = len(x)
            ny = len(y)
            if nx != ny:
                raise ValueError("Length mismatch: len(x)="+repr(nx)+", len(y)="+repr(ny))
            else:
                z = np.zeros(nx)
                for i in range(nx):
                    z[i] = self.topography_interpolated(x[i],y[i])[0]
                return z
        else:
            return self.topography_interpolated(x,y)[0][0]