import dolfin
import dolfin_adjoint
from windse import windse_parameters
import numpy as np

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

    The original function doesn't account for kwargs in a project so it 
    doesn't pass them to this function. For now, we just supply the
    solver_parameters manually in this case. 

    Todo:

        Eventually, we want to replace this with a full PetscKrylovSolver()
        to get access to all the ksp options.

    """
    eq = prepared[0]
    func = prepared[1]
    bcs = prepared[2]


    if not self.forward_kwargs:

        # print()
        # print()
        # print()

        # print(dir(eq.rhs.coefficients()[-1]))
        # print(eq.rhs.coefficients()[-1].values())
        # print(eq.rhs.coefficients()[-1].name())
        # for i in range(len(eq.rhs.coefficients())):
        #     print(type(eq.rhs.coefficients()[i]))
        #     if "constant" in str(type(eq.rhs.coefficients()[i])).lower():
        #         print(eq.rhs.coefficients()[i].values())
        #         print(eq.rhs.coefficients()[i].name())
        # print()
        # print()
        # print()

        dolfin_adjoint.backend.solve(eq, func, bcs, solver_parameters={'linear_solver': 'mumps'})
    else:
        dolfin_adjoint.backend.solve(eq, func, bcs, **self.forward_kwargs)
    return func

dolfin_adjoint.solving.SolveBlock.recompute_component = recompute_component

def update_relative_heights(tape):
    """This function find the the turbine (x,y,z) control values and updates
    the z values according to the updated (x,y) values that are being
    optimized. 

    """

    ### This gets the list of Constants associated with the turbine force ###
    blocks = tape.get_blocks()
    depends = blocks[0].get_dependencies()

    ### This loops over the Constants and identifies them ###
    x_ind = []
    y_ind = []
    z_ind = []
    for i in range(len(depends)):
        cur = depends[i]

        # print(cur.output)
        # print(cur.saved_output)

        if "x" in str(cur.output):
            x_ind.append(i)
            # x_ind.append(str(cur.saved_output))
        if "y" in str(cur.output):
            y_ind.append(i)
            # y_ind.append(str(cur.saved_output))
        if "z" in str(cur.output):
            z_ind.append(i)
            # z_ind.append(str(cur.saved_output))

    # print(x_ind)
    # print(y_ind)
    # print(z_ind)
    # print()
    # print()

    ### Finally we extract the x and y values and update the z values ###
    for i in range(len(z_ind)):
        x_val = depends[x_ind[i]].saved_output.values()[0]
        y_val = depends[y_ind[i]].saved_output.values()[0]
        z_val = float(windse_parameters.ground_fx(x_val,y_val)) + windse_parameters.full_hh[i]
        depends[z_ind[i]].saved_output.assign(z_val)
        print(x_val,y_val,z_val)

    # exit()


def reduced_functional_eval(self, values):
    """This function overrides 
    pyadjoint.reduced_functional.ReducedFunctional.__call__() allowing
    windse to update the turbine absolute heights after the locations
    are updated.

    Args:
        values ([OverloadedType]): If you have multiple controls this 
            should be a list of new values for each control in the order
            you listed the controls to the constructor. If you have a 
            single control it can either be a list or a single object. 
            Each new value should have the same type as the 
            corresponding control.

    Returns:
        :obj:`OverloadedType`: The computed value. Typically of instance
            of :class:`AdjFloat`.

    """
    values = dolfin_adjoint.pyadjoint.enlisting.Enlist(values)
    if len(values) != len(self.controls):
        raise ValueError("values should be a list of same length as controls.")

    # Call callback.
    self.eval_cb_pre(self.controls.delist(values))

    for i, value in enumerate(values):
        self.controls[i].update(value)

    ### This is the new code injected into pyadjoint ###
    update_relative_heights(self.tape)
    ####################################################

    blocks = self.tape.get_blocks()
    with self.marked_controls():
        with dolfin_adjoint.pyadjoint.tape.stop_annotating():
            for i in range(len(blocks)):
                blocks[i].recompute()

    func_value = self.functional.block_variable.checkpoint

    # Call callback
    self.eval_cb_post(func_value, self.controls.delist(values))

    return func_value

dolfin_adjoint.ReducedFunctional.__call__ = reduced_functional_eval