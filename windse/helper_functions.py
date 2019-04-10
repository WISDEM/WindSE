from dolfin import Constant

def BaseHeight(x,y,ground):
    return Constant(ground(float(x),float(y)))


