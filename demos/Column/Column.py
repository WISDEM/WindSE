import windse
import numpy as np

parameters['form_compiler']['quadrature_degree'] = 6

### Initialize WindSE ###
windse.initialize("params.yaml")

### Generate Domain ###
dom = windse.BoxDomain()

### Generate Wind Farm ###
farm = windse.ImportedWindFarm(dom)
farm.Plot()

### Warp the mesh and refine ###
dom.Warp(80,80,0.75)
# region = [[farm.ex_x[0],dom.x_range[1]],farm.ex_y,farm.ex_z]
dom.Refine(1,local=True)
dom.Save()

### Function Space ###
fs = windse.LinearFunctionSpace(dom)

### Setup Boundary Conditions ###
bc = windse.LinearInflow(dom,fs)

### Generate the problem ###
problem = windse.StabilizedProblem(dom,farm,fs,bc)

### Solve ###
solver = windse.SteadySolver(problem)
solver.Solve()

### Output Results ###
solver.Save()
