
__author__ = "Ryan King <ryan.king@nrel.gov>"
__date__ = "2017-11-21"
__copyright__ = "Copyright (c) NREL 2017. All rights reserved."
__license__  = "Apache License, Version 2.0"


from dolfin import *
import numpy as np
import random as random
from scipy import integrate
from scipy import stats
import time
import sys
sys.getrecursionlimit()
sys.setrecursionlimit(10000)

set_log_level(INFO)
# mpirun -np 4 python windse4dir.py 2>&1 | tee log
'''
CRITICAL  = 50, // errors that may lead to data corruption and suchlike
ERROR     = 40, // things that go booms
WARNING   = 30, // things that may go boom later
INFO      = 20, // information of general interest
PROGRESS  = 16, // what's happening (broadly)
TRACE     = 13, // what's happening (in detail)
DBG       = 10  // sundry

'''

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;
parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -fno-math-errno -march=native'        
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']['quadrature_degree'] = 12

#actual turbine diam in meters
RD = 80.

# mesh parameters
degree = 2
Lx = 60.*RD
Ly = 60.*RD
nx = 50
ny = 50

#WTG parameters
numturbs = 16

inflowVel=8
#number of inflow direction bins
bins = 1
WTGexp = 8.
radius = RD/2.
thickness = RD/20.
numRefine = 2
A=RD # weird for 2D
HH=80    
initExtent=.95
mlDenom=2.


#site/refinement constraints
site_x = 15.*RD
site_y = 15.*RD

restart = False
randStart = False
gridStart = True
optimize = False

if optimize == True:
    from dolfin_adjoint import *


# inflow is always from left in simulation
dirs = np.linspace(pi/2., 2*pi+pi/2., bins, endpoint = False)
weights = np.ones(bins)/bins

#which inflow angle to plot
alpha = dirs[0]

#domain centered on (0,0)
mesh = RectangleMesh(Point(-Lx/2., -Ly/2.), Point(Lx/2., Ly/2.), nx, ny)

# actuator disk distribution for normalization constant
def WTGdist(x,y):
    return np.exp(-((x/thickness)**WTGexp + (y/radius)**WTGexp))

def refine_mesh(mesh, site_x, site_y, HH):
    #refines the mesh around the site boundaries
    h = mesh.hmin()
    
    cell_f = CellFunction('bool', mesh, False)
    for cell in cells(mesh):
        if (cell.midpoint()[0]**2 + cell.midpoint()[1]**2 < site_x**2 + site_y**2 + h) :
            cell_f[cell] = True

    mesh = refine(mesh, cell_f)

    return mesh

h = mesh.hmin()

for nums in range(numRefine):
    # print 'refining mesh'
    mesh=refine_mesh(mesh, site_x, site_y, HH)
    h = mesh.hmin()

# function spaces, mixed function space syntax not backwards compatible
V = VectorElement('Lagrange', mesh.ufl_cell(), 2) 
Q = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
VQ = FunctionSpace(mesh, MixedElement([V,Q]))   #NSE equations
V = VectorFunctionSpace(mesh, 'Lagrange', 2)
Q = FunctionSpace(mesh, 'Lagrange', 1)


def createLayout(numturbs):
    mx=[]
    my=[]
    mz=[]

    if randStart == True:
        for i in range(numturbs):
            mx.append(Constant(np.random.uniform(low=-(site_x - radius),high=(site_x - radius))))
            my.append(Constant(np.random.uniform(low=-(site_y - radius), high=(site_y - radius))))
            mz.append(Constant(HH)) 

    elif gridStart ==True:

        if numturbs == 16:
            rows = 4
            cols = 4
            xpos = np.linspace(-initExtent*(site_x - radius),initExtent*(site_x - radius),cols)
            ypos = np.linspace(-initExtent*(site_y - radius),initExtent*(site_y - radius),rows)
            for i in range(rows):
                for j in range(cols):
                    mx.append(Constant(xpos[j]))
                    my.append(Constant(ypos[i]))
                    # # some starting noise sometimes helps
                    # mx.append(Constant(xpos[j]+5.*np.random.randn()))
                    # my.append(Constant(ypos[i]+5.*np.random.randn()))
                    mz.append(Constant(HH))


    elif restart == True:
        # fixed layout here
        m_temp = [Constant(-113.961988283),Constant(-386.535837904),Constant(-512.116113959),Constant(-237.354391531),Constant(638.697968355),Constant(13.6826901448),Constant(386.535838424),Constant(-113.961987466),Constant(13.6826875361),Constant(-638.697971072),Constant(-887.942379804),Constant(-813.542880381),Constant(813.542880031),Constant(-887.942379852),Constant(237.354391629),Constant(-512.116113931),Constant(-237.3543916),Constant(512.116113865),Constant(-813.542880345),Constant(887.942379783),Constant(887.942379753),Constant(813.542880265),Constant(-13.6826884631),Constant(638.697970038),Constant(-386.535837846),Constant(113.961988218),Constant(-638.697970958),Constant(-13.6826879195),Constant(512.116113711),Constant(237.354391612),Constant(113.961988),Constant(386.535838129)]
        
        mx,my = splitSolution(m_temp,numturbs)
        for i in range(numturbs):
            mz.append(Constant(HH))

    return mx, my, mz

def createRotatedTurbineForce(mx,my,ma,A,beta,numturbs,alpha,V):
    x=SpatialCoordinate(mesh)
    WTGbase = project(Expression(("1.0","0.0"),degree=2),V)
    tf = Function(V)

    for i in range(numturbs):
        #rotation
        xrot = cos(alpha)*mx[i] - sin(alpha)*my[i]
        yrot = sin(alpha)*mx[i] + cos(alpha)*my[i]

    
        tf = tf + 0.5*4.*A*ma[i]/(1.-ma[i])/beta*exp(-(((x[0] - xrot)/thickness)**WTGexp + ((x[1] - yrot)/radius)**WTGexp))*WTGbase.copy(deepcopy=True)

    return tf

def rotatedPowerFunctional(alpha,A,beta,mx,my,ma,u,numturbs,V):
    # functional for dolfin-adjoint
    x=SpatialCoordinate(mesh)
    J=Functional(0.)
    for i in range(numturbs):
        #rotation
        xrot = cos(alpha)*mx[i] - sin(alpha)*my[i]
        yrot = sin(alpha)*mx[i] + cos(alpha)*my[i]

        J = J + 0.5*4.*np.pi*radius**2*ma[i]/(1.-ma[i])*Functional(exp(-(((x[0] - xrot)/thickness)**WTGexp + ((x[1] - yrot)/radius)**WTGexp))*u[0]**3.*dx)

    return J

def rotatedPowerFunction(alpha,A,beta,mx,my,ma,up,numturbs,V):
    #emulating an actual power curve
    x=SpatialCoordinate(mesh)

    for i in range(numturbs):
        #rotation
        xrot = cos(alpha)*mx[i] - sin(alpha)*my[i]
        yrot = sin(alpha)*mx[i] + cos(alpha)*my[i]

        if i ==0:
            J = 0.5*np.pi*radius**2*4*float(ma[i])*(1.-float(ma[i]))**2*up.sub(0)(xrot,yrot)[0]**3
        else:
            J = J + 0.5*np.pi*radius**2*4*float(ma[i])*(1.-float(ma[i]))**2*up.sub(0)(xrot,yrot)[0]**3

    return J


def createControl(mx,my,numturbs):
        m = []
        for i in range(numturbs):
            m.append(Control(mx[i]))
            m.append(Control(my[i]))
            
        return m

def createBounds(mx,my,numturbs):
        ub=[]
        lb=[]
        for i in range(numturbs):
            lb.append(Constant(-(site_x - radius)))
            lb.append(Constant(-(site_y - radius)))
            ub.append(Constant((site_x - radius)))
            ub.append(Constant((site_y - radius)))
            
        bounds = [lb,ub]
        return bounds

def createAxialBounds(ma,numturbs):
        ub=[]
        lb=[]
        for i in range(numturbs):
            lb.append(Constant(0.))
            ub.append(Constant(0.75))
            
        bounds = [lb,ub]
        return bounds

def splitSolution(m_opt,numturbs):
    mx_opt = []
    my_opt = []
    j=0
    for i in range(numturbs):
        mx_opt.append(m_opt[j])
        j+=1
        my_opt.append(m_opt[j])
        j+=1
        
    return mx_opt,my_opt

def main(tf):
    nu = Constant(.00005)
    f = Constant((0.,0.))
    up_next = Function(VQ)
    u_next,p_next = split(up_next)
    v,q = TestFunctions(VQ)
    class InitialConditions(Expression):
        def __init__(self,**kwargs):
            random.seed(2)
        def eval(self, values, x):
            values[0] = inflowVel
            values[1] = 0.0
            values[2] = 0.0
        def value_shape(self):
            return (3,)  
    #boundary conditions
    class NoSlipBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1]**2 - (Ly/2.)**2, 0.) and on_boundary

    class InflowBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0],-(Lx/2.)) and on_boundary

    class PeriodicBoundary(SubDomain):

        def inside(self, x, on_boundary):
            # return True if on left or bottom boundary AND NOT on one of the two slave edges
            return bool((near(x[0], -(Lx/2.)) or near(x[1], -(Ly/2.))) and 
                    (not (near(x[0], (Lx/2.)) or near(x[1], (Ly/2.)))) and on_boundary)
                          
        def map(self, x, y):
            if near(x[0], (Lx/2.)) and near(x[1], (Ly/2.)):
                y[0] = x[0] - 2*(Lx/2.)
                y[1] = x[1] - 2*(Ly/2.)
            elif near(x[0], (Lx/2.)):
                y[0] = x[0] - 2*(Lx/2.)
                y[1] = x[1]
            else: # near(x[1], (Ly/2.)):
                y[0] = x[0]
                y[1] = x[1] - 2*(Ly/2.)

    u0 = InitialConditions(degree=2)  
    up_next.interpolate(u0)

    lmix = radius/mlDenom

    S = sqrt(2.*inner(0.5*(grad(u_next)+grad(u_next).T),0.5*(grad(u_next)+grad(u_next).T)))
    nu_T=lmix**2.*S

    F = inner(grad(u_next)*u_next, v)*dx + (nu+nu_T)*inner(grad(u_next), grad(v))*dx - inner(div(v),p_next)*dx - inner(div(u_next),q)*dx - inner(f,v)*dx + inner(tf*u_next[0]**2,v)*dx 

    # lateral BC
    # bc1 = DirichletBC(VQ.sub(0), Constant((8.0,0.0)), NoSlipBoundary())
    bc1a = DirichletBC(VQ.sub(0).sub(1), Constant(0.0), NoSlipBoundary())

    # inflow BC
    bc2 = DirichletBC(VQ.sub(0), Constant((inflowVel,0.0)), InflowBoundary())
    # bc2a = DirichletBC(VQ.sub(0).sub(0), Constant(8.), InflowBoundary())

    # outflow pressure BC is implicitly 0

    bc = [bc1a,bc2]

    solve(F == 0, up_next, bc, solver_parameters={"newton_solver":{"absolute_tolerance": 1e-8}})
    u_next,p_next = split(up_next)

    if optimize == False:
        nu_T_out=project(nu_T, Q)
        lStr= 'nu_t.pvd'
        file = File(lStr)
        file << nu_T_out
    return u_next, up_next

if __name__ == "__main__":
    
    mx,my,mz = createLayout(numturbs)

    ma=[Constant(mm) for mm in 0.33*np.ones(numturbs)]

    beta = integrate.dblquad(WTGdist,-3*radius,3*radius,lambda x: -3*radius,lambda x: 3*radius)

    B=beta[0]

    if optimize == True: 

        # power functional
        J = Functional(0.)
        for i in range(bins):
            tf_rot= createRotatedTurbineForce(mx,my,ma,A,B,numturbs,dirs[i],V)
            u_rot, p_rot = main(tf_rot)           
            J = J + weights[i]*rotatedPowerFunctional(dirs[i],A,B,mx,my,ma,u_rot,numturbs,V)

        # position control variables
        m=createControl(mx,my,numturbs)
        bounds = createBounds(mx,my,numturbs)

        rf = ReducedFunctional(J,m)

        def iter_cb(m):
            if MPI.rank(mpi_comm_world()) == 0:
                print("m = ")
                for mm in m:
                    print("Constant("+ str(mm)+ "),")
        m_opt = maximize(rf, method="L-BFGS-B", options = {"disp": True}, bounds = bounds, callback = iter_cb)
        # m_opt = maximize(rf, method="SLSQP", options = {"disp": True}, bounds = bounds, callback = iter_cb)
        mx_opt,my_opt=splitSolution(m_opt,numturbs)
        mz_opt=mz

    # otherwise go strait to final plotting routine, plot last inflow direction
    else:
        mx_opt = mx
        my_opt = my
        mz_opt = mz

    tf_rot_opt= createRotatedTurbineForce(mx_opt,my_opt,ma,A,B,numturbs,dirs[0],V)
    tf_rot_opt_out = project(tf_rot_opt, V)
    # plot(tf_rot_opt_out)
    tfStr='tf.pvd' 
    file1 = File(tfStr)
    file1 << tf_rot_opt_out

    u_rot_opt,up_rot_opt  = main(tf_rot_opt)

    for i in range(bins):
        
        # report power curve scalar function instead of functional for adjoint derivatives
        if i ==0:
            Jfunc = weights[i]*rotatedPowerFunction(dirs[i],A,B,mx_opt,my_opt,ma,up_rot_opt,numturbs,V)
        else:
            Jfunc = Jfunc + weights[i]*rotatedPowerFunction(dirs[i],A,B,mx_opt,my_opt,ma,up_rot_opt,numturbs,V)
    # power = assemble(Jfunc)
    print(Jfunc)
    u_rot_opt_out=project(u_rot_opt, V)

    uStr='u.pvd'
    file2 = File(uStr)
    file2 << u_rot_opt_out
