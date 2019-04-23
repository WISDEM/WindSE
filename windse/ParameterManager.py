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
    from dolfin import File, HDF5File, XDMFFile, MPI, Mesh

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
        self.current_tab = 0

    def Load(self, loc):
        """
        This function loads the parameters from the .yaml file. 
        It should only be assessed once from the :meth:`windse.initialize` function.

        Args:
            loc (str): This string is the location of the .yaml parameters file.

        """

        ### Load the yaml file (requires PyYaml)
        self.fprint("Loading and Setting up Parameters", special="header")
        self.update(yaml.load(open(loc)))

        ### Create Instances of the general options ###
        self.name = self["general"].get("name", "Test")
        self.preappend_datetime = self["general"].get("preappend_datetime", False)
        self.save_file_type = self["general"].get("save_file_type", "pvd")
        self.dolfin_adjoint = self["general"].get("dolfin_adjoint", False)
        self.outputs = self["general"].get("outputs", [])

        ### Print some stats ###
        self.fprint("Run Name: {0}".format(self.name), offset=1)

        ### Set up the folder Structure ###
        timestamp=datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
        fancytimestamp=datetime.datetime.today().strftime('%Y/%m/%d_%H:%M:%S')
        if self.preappend_datetime:
            self.name = timestamp+"-"+self.name
            self["general"]["name"]=self.name
        self.folder = "output/"+self.name+"/"
        self["general"]["folder"] = self.folder

        ### Create checkpoint if required ###
        # if self.save_file_type == "hdf5":
        #     self.Hdf=HDF5File(MPI.mpi_comm(), self.folder+"checkpoint/checkpoint.h5", "w")

        ### Print some more stuff
        self.fprint("Run Time Stamp: {0}".format(fancytimestamp), offset=1)
        self.fprint("Output Folder: {0}".format(self.folder), offset=1)
        self.fprint("Parameters Setup", special="footer")

    def Read(self):
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

    def Save(self, func, filename, subfolder="",val=0,file=None,filetype="default"):
        """
        This function is used to save the various dolfin.Functions created
        by windse. It should only be accessed internally.

        Args:
            func (dolfin.Function): The Function to be saved
            filename (str): the name of the function

        :Keyword Arguments:
            * **subfolder** (*str*): where to save the files within the output folder
            * **n** (*float*): used for saving a series of output. Use n=0 for the first save.

        """
        self.fprint("Saving: {0}".format(filename), offset=1)

        ### Name the function in the meta data, This should probably be done at creation
        func.rename(filename,filename)

        if filetype == "default":
            filetype = self.save_file_type

        if file is None:
            ### Make sure the folder exists
            if not os.path.exists(self.folder+subfolder): os.makedirs(self.folder+subfolder)

            if filetype == "pvd":
                file_string = self.folder+subfolder+filename+".pvd"
                out = File(file_string)
                out << (func,val)
            elif filetype == "xdmf":
                file_string = self.folder+subfolder+filename+".xdmf"
                out = XDMFFile(file_string)
                out.write(func,val)

            return out

        else:
            if filetype == "pvd" or isinstance(func,type(Mesh)):
                file << (func,val)
            elif filetype == "xdmf":
                file.write(func,val)

    def fprint(self,string,tab=None,offset=0,special=None):
        """
        This is just a fancy print function that will tab according to where
        we are in the solve

        Args:
            string (str): the string for printing

        :Keyword Arguments:
            * **tab** (*int*): the tab level

        """
        if tab is None:
            tab = self.current_tab
        
        tab += offset 

        if special=="header":
            self.fprint("",tab=tab)
        elif special =="footer":
            self.fprint("",tab=tab+1)

        rank = 0
        if rank == 0:
            tabbed = "|    "*tab
            if isinstance(string,str):
                string = tabbed+string
            else:
                string = tabbed+repr(string)
            print(string)
            # sys.stdout.flush()

        if special=="header":
            self.fprint("",tab=tab+1)

windse_parameters = Parameters()