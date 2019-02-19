### This won't be needed once the package is part of local python path ###
import sys
sys.path.insert(0, '../../dev/')


from dolfin import *
from dolfin_adjoint import *
import WindSE
import numpy as np

parameters['form_compiler']['quadrature_degree'] = 6

### Create an Instance of the Options ###
# options = WindSE.Options("params.yaml")

### Generate Domain ###
dom = WindSE.BoxDomain()

### Generate Wind Farm ###
farm = WindSE.GridWindFarm( dom)
farm.Plot(False)

### Warp the mesh and refine ###
dom.Warp(200,0.75)
# dom.WarpNonlinear(1.8)
region = [[farm.ex_x[0],dom.x_range[1]],farm.ex_y,farm.ex_z]
dom.Refine(1,region=region)
dom.Save()


print(len(dom.mesh.coordinates()[:]))
print(len(farm.dom.mesh.coordinates()[:]))
print(print(farm.dom.mesh.hmin()))
# exit()

### Function Space ###
fs = WindSE.LinearFunctionSpace(dom)

print(fs.Q.dim())

### Setup Boundary Conditions ###
bc = WindSE.LinearInflow(dom,fs)

### Generate the problem ###
problem = WindSE.StabilizedProblem(dom,farm,fs,bc)

### Solve ###
solver = WindSE.DefaultSolver(problem)
solver.Solve()

# control = WindSE.CreateControl(farm.ma)
# bounds = WindSE.CreateAxialBounds(farm.ma)

# J=WindSE.PowerFunctional(problem.tf,solver.u_next)
# rf=ReducedFunctional(J,control)
# # print(J)
# # print(float(J))
# # dJdma= compute_gradient(J, control)
# # print([float(dd) for dd in dJdma])

# def iter_cb(m):
# 	# if MPI.rank(mpi_comm_world()) == 0:
# 	print("m = ")
# 	for mm in m:
# 		print("Constant("+ str(mm)+ "),")

# m_opt=minimize(rf, method="L-BFGS-B", options = {"disp": True}, bounds = bounds, callback = iter_cb)
# print([float(mm) for mm in m_opt])
# farm.ma = m_opt
# solver.Solve()

# h = [Constant(0.001),Constant(0.001)]  # the direction of the perturbation
# Jhat = ReducedFunctional(J, control)  
# conv_rate = taylor_test(Jhat, farm.ma, h)
# print(conv_rate)

### Output Results ###
solver.Save()
