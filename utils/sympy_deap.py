import sympy as sp
from utils.sympy_prefix import operators, numbers_types

deap_to_sympy = {
    'add': sp.Add,
    'sub': lambda x, y: sp.Add(x, -y),
    'mul': sp.Mul,
    'div': lambda x, y: sp.Mul(x, sp.Pow(y, -1)),
    'pow': sp.Pow,
    'sin': sp.sin,
    'cos': sp.cos,
    'tan': sp.tan,
    'abs': sp.Abs,
    'max': sp.Max,
    'min': sp.Min,
    'sinh': sp.sinh,
    'cosh': sp.cosh,
    'tanh': sp.tanh,
    'asin': sp.asin,
    'acos': sp.acos,
    'atan': sp.atan,
    'protected_div': lambda x, y: sp.Mul(x, sp.Pow(y, -1)) if y != 0 else 1,
    'protected_pow': sp.Pow,
    'protected_exp': sp.exp,
    'protected_log': lambda x : sp.log(x),
    'protected_sqrt': lambda x : sp.sqrt(x),
    'pi': sp.pi,
    'neg': lambda x: -x
}

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


def deap_to_sympy(func_form):
    """
    Convert a functional form string back into a SymPy expression.
    
    Parameters:
    - func_form: String in functional form (e.g., 'mul(s_1, s_2, s_4, pow(s_3, -1))')
    
    Returns:
    - SymPy expression.
    """
    def _parse_expr(expression):
        for func, sympy_func in deap_to_sympy.items():
            if expression.startswith(f"{func}(") and expression.endswith(")"):
                args_str = expression[len(func) + 1:-1]
                args = _split_args(args_str)
                parsed_args = [_parse_expr(arg) for arg in args]
                return sympy_func(*parsed_args)
        
        if expression.startswith('s_'):
            return sp.Symbol(expression)
        
        try:
            return sp.sympify(expression)
        except sp.SympifyError:
            raise ValueError(f"Unsupported expression format: {expression}")

    def _split_args(args_str):
        """
        Split the argument string into individual arguments, considering nested functions.
        """
        args = []
        current_arg = []
        depth = 0

        for char in args_str:
            if char == ',' and depth == 0:
                args.append(''.join(current_arg).strip())
                current_arg = []
            else:
                current_arg.append(char)
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1

        args.append(''.join(current_arg).strip())
        return args

    return _parse_expr(func_form) 
