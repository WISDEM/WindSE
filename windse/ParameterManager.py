"""
The ParameterManager controls the handles importing 
the parameters from the params.yaml file. These
functions don't need to be accessed by the end user.
"""

import __main__
import os

### Get the name of program importing this package ###
main_file = os.path.basename(__main__.__file__)

### This checks if we are just doing documentation ###
if main_file != "sphinx-build":
    import yaml
    import datetime
    import numpy as np
    from math import ceil
    from dolfin import File, HDF5File, XDMFFile, MPI

######################################################
### Collect all options and define general options ###
######################################################



class Parameters(dict):
    """
    Parameters is a subclass of pythons *dict* that adds
    function specific to windse.
    """
    def __init__(self):
        super(Parameters, self).__init__()

    def Load(self, loc):
        """
        This function loads the parameters from the .yaml file. 
        It should only be assessed once from the :meth:`windse.initialize` function.

        Args:
            loc (str): This string is the location of the .yaml parameters file.

        """

        ### Load the yaml file (requires PyYaml)
        self.update(yaml.load(open(loc)))

        ### Create Instances of the general options ###
        self.name = self["general"].get("name", "Test")
        self.preappend_datetime = self["general"].get("preappend_datetime", False)
        self.save_file_type = self["general"].get("save_file_type", "pvd")
        self.dolfin_adjoint = self["general"].get("dolfin_adjoint", False)

        ### Set up the folder Structure ###
        if self.preappend_datetime:
            self.name = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')+"-"+self.name
            self["general"]["name"]=self.name
        self.folder = "output/"+self.name+"/"
        self["general"]["folder"] = self.folder

        ### Create checkpoint if required ###
        # if self.save_file_type == "hdf5":
        #     self.Hdf=HDF5File(MPI.mpi_comm(), self.folder+"checkpoint/checkpoint.h5", "w")

    def Print(self):
        """
        This function reads the current state of the parameters object 
        and prints it in a easy to read way.
        """
        for group in self:
            print(group)
            max_length = 0
            for key in self[group]:
                max_length = max(max_length,len(key))
            max_length = max_length
            for key in self[group]:
                print("    "+key+":  "+" "*(max_length-len(key))+repr(self[group][key]))

    def Save(self, file, filename, subfolder="",t=0,reuse=None):
        """
        This function is used to save the various dolfin.Functions created
        by windse. It should only be accessed internally.

        Args:
            file (dolfin.Function): The Function to be saved
            filename (str): the name of the function

        :Keyword Arguments:
            * **subfolder** (*str*): where to save the files within the output folder

        """
        print("Saving "+filename)

        ### Name the file in the meta data
        file.rename(filename,filename)

        if reuse is None:
            ### Make sure the folder exists
            if not os.path.exists(self.folder+subfolder): os.makedirs(self.folder+subfolder)

            if self.save_file_type == "pvd":
                file_string = self.folder+subfolder+filename+".pvd"
                out = File(file_string)
                out << (file,t)
            elif self.save_file_type == "xdmf":
                file_string = self.folder+subfolder+filename+".xdmf"
                out = XDMFFile(file_string)
                out.write(file,t)
            print(filename+" Saved")

            return out

        else:
            if self.save_file_type == "pvd":
                reuse << (file,t)
            elif self.save_file_type == "xdmf":
                reuse.write(file,t)
            print(filename+" Saved")


windse_parameters = Parameters()