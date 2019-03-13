import dolfin_adjoint

def linalg_solve(*args, **kwargs):
    """This function overrides dolfin_adjoints.compat.linalg_solve.

    The original function doesn't allow for solver options because it uses
     the::

        dolfin.solve(A,x,b) 

    form which doesn't accept keyword arguments. However, It does except 
    additional arguments that defined some solver options, which we pass
    in manually

    Todo:

        Eventually, we want to replace this with a full PetscKrylovSolver()
        to get access to all the ksp options.

    """

    return dolfin_adjoint.backend.solve(*args,"mumps") 

dolfin_adjoint.types.compat.linalg_solve = linalg_solve


def recompute_component(self, inputs, block_variable, idx, prepared):
    """This function overrides 
    dolfin_adjoint.solving.SolveBlock.recompute_component

    The original function doesn't set parameters for solving a linear
    problem if the forward problem is nonlinear. For now, we just supply the
    solver_parameters manually in this case. 

    Todo:

        Eventually, we want to replace this with a full PetscKrylovSolver()
        to get access to all the ksp options.

    """
    eq = prepared[0]
    func = prepared[1]
    bcs = prepared[2]

    if not self.forward_kwargs:
        dolfin_adjoint.backend.solve(eq, func, bcs, solver_parameters={'linear_solver': 'mumps'})
    else:
        dolfin_adjoint.backend.solve(eq, func, bcs, **self.forward_kwargs)
    return func

dolfin_adjoint.solving.SolveBlock.recompute_component = recompute_component