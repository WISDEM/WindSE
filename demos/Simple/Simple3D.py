import windse

### Initialize WindSE ###
windse.initialize("params.yaml")

### Generate Domain ###
dom = windse.BoxDomain()
dom.Save()

### Generate Wind Farm ###
farm = windse.RandomWindFarm(dom)
farm.Plot()

### Function Space ###
fs = windse.LinearFunctionSpace(dom)

### Setup Boundary Conditions ###
bc = windse.LinearInflow(dom,fs)

### Generate the problem ###
problem = windse.StabilizedProblem(dom,farm,fs,bc)

### Solve ###
solver = windse.DefaultSolver(problem)
solver.Solve()

### Output Results ###
solver.Save()
solver.Plot()