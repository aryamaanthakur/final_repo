import sympy as sp
from sympy_prefix import operators, numbers_types

def sympy_to_deap(expr):
   
    for key,value in operators.items():
        if isinstance(expr, key):
            args = expr.args
            return f"{value}({', '.join(sympy_to_deap(arg) for arg in args)})"
        
    for item in numbers_types:
        if type(expr) == sp.core.numbers.Rational or type(expr) == sp.core.numbers.Float:
            return f"div({str(sp.Rational(expr).p)}, {str(sp.Rational(expr).q)})"
        elif type(expr) == item:
            return str(expr)
        
    if isinstance(expr, sp.Symbol):
        return str(expr)
    else:
        print(expr)
        raise ValueError(f"Unsupported expression type: {type(expr)}")
    
def deap_to_sympy(expr):
    pass