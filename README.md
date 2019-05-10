## DEPRECATED 

THIS REPOSITORY IS DEPRECATED AND WILL BE DELETED AT THE END OF 2019. UNTIL THEN, IT WILL BE KEPT IN READ-ONLY MODE AND NO LONGER MAINTAINED.

The active WindSE can be found at https://github.com/NREL/WindSE/

# WindSE: Wind Systems Engineering

## Simple Description:

WindSE is a python package that uses a FEniCS backend to perform wind farm simulations and optimization.  

## Installation:

### General Installation:

In order to use WindSE, the python version fenics 2018.1.0 or later must be installed along with a compatible version of dolfin-adjoint. WindSE can be installed by downloading the source files from the GitHub (https://github.com/WISDEM/WindSE) and running the command: 
```
pip install -e .
```
in the root folder. 

### Conda Installation (Automatic):

The best way to use WindSE is via a conda environment. To install conda check out this link: [Conda Installation.](https://conda.io/projects/conda/en/latest/user-guide/install/) After conda is installed, you need to download/clone the WindSE repo. Then can automatically setup the WindSE environment using:
```
conda env create -f environment.yml
```
in the root folder (where environment.yml is located). Next we activate the environment using:
```
source activate fenics_windse
```
Then create a link to the WindSE package by running:
```
pip install -e .
```
in the same folder.

### Conda Installation (Manual):

If the automatic setup fails, or you just want to setup the environment manually, follow these steps. After conda is installed, create a new environment using:
```
conda create --name fenics_windse
```
You can replace the name "fenics_windse" with a different name for the environment if you want. Next we activate the environment using:
```
source activate fenics_windse
```
or whatever you named your environment. Now we need to install the dependent packages using:
```
conda install -c conda-forge fenics=2018.1.0 dolfin-adjoint mshr matplotlib scipy pyyaml
```
Finally, download/clone the WindSE repo and run:
```
pip install -e .
```
in the root folder. 

