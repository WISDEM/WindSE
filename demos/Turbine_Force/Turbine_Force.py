from dolfin import *
import windse
import numpy as np

parameters['form_compiler']['quadrature_degree'] = 6

### Initialize WindSE ###
windse.initialize("params.yaml")

### Generate Domain ###
dom = windse.BoxDomain()
dom.Save()

### Generate Wind Farm ###
farm = windse.ImportedWindFarm(dom)
farm.Plot()

### Function Space ###
fs = windse.LinearFunctionSpace(dom)

### Save the Turbine Force ##
# farm.RotateFarm(3.14159/4.0)
u = farm.ModTurbineForce(fs.Q,dom.mesh)

print("Starting Project")
u = project(u,fs.Q,solver_type='mumps')
u.rename("force","force")
print("Done Project")

filename = windse.windse_parameters.folder+"/functions/turbine_force.pvd"
File(filename) << u