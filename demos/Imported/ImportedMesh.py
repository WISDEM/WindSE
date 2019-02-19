import windse

### Initialize WindSE ###
windse.initialize("params.yaml")

### Generate Domain ###
dom = windse.ImportedDomain()

### Generate Wind Farm ###
farm = windse.GridWindFarm(dom)
farm.Plot()

### Refine the Domain ###
region = [[farm.ex_x[0],dom.x_range[1]],farm.ex_y,farm.ex_z]
dom.Refine(1,region=region)
dom.Save()

### Function Space ###
fs = windse.LinearFunctionSpace(dom)
print(fs.W.dim())

### Setup Boundary Conditions ###
bc = windse.LinearInflow(dom,fs)

### Generate the problem ###
problem = windse.StabilizedProblem(dom,farm,fs,bc)

### Solve ###
solver = windse.DefaultSolver(problem)
solver.Solve()

### Output Results ###
solver.Save()