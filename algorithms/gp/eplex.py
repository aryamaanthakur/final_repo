import os
import pickle
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional
from sympy import sympify, sin, cos, tan
from ellyn import ellyn

@dataclass
class EplexConfig:
    selection: Optional[str] = field(default='epsilon_lexicase')
    lex_eps_global: Optional[bool] = field(default=False)
    lex_eps_dynamic: Optional[bool] = field(default=False)
    islands: Optional[bool] = field(default=False)
    num_islands: Optional[int] = field(default=10)
    island_gens: Optional[int] = field(default=100)
    verbosity: Optional[int] = field(default=0)
    print_data: Optional[bool] = field(default=False)
    elitism: Optional[bool] = field(default=True)
    pHC_on: Optional[bool] = field(default=True)
    prto_arch_on: Optional[bool] = field(default=True)
    max_len: Optional[int] = field(default = 64)
    max_len_init: Optional[int] = field(default=20)
    popsize: Optional[int] = field(default=500)
    g: Optional[int] = field(default=500)
    time_limit: Optional[int] = field(default=2*60)

    @classmethod
    def from_yaml(cls, yaml_string, from_file=True):
        if from_file:
            with open(yaml_string) as f:
                yaml_string = f.read()
        
        args_dict = yaml.safe_load(yaml_string)
        try:
            config = EplexConfig(**args_dict)
        except TypeError:
            ExtendedEplexConfig = dataclass(type("ExtendedEplexConfig", (object,), {"__annotations__": args_dict}))
            config = ExtendedEplexConfig(**args_dict)
        return config

class EplexRegressor:
    def __init__(self, config):
        self.config = config
        self.converter = {'sub': lambda x, y : x - y,
                          'div': lambda x, y : x/y,
                          'mul': lambda x, y : x*y,
                          'add': lambda x, y : x + y,
                          'neg': lambda x    : -x,
                          'pow': lambda x, y : x**y,
                          'sin': lambda x    : sin(x),
                          'cos': lambda x    : cos(x),
                          'tan': lambda x    : tan(x),
                          'inv': lambda x: 1/x,
                          'sqrt': lambda x: x**0.5,
                          'pow3': lambda x: x**3,
                          }

    def get_model(self):
        model = ellyn(**asdict(self.config))
        return model
    
    def predict_single(self, X, y, sympify_expr=True):
        self.model = self.get_model()
        self.model.fit(X, y)
        eqn = self.model.stack_2_eqn(self.model.best_estimator_)
        if sympify_expr:
            return sympify(eqn, locals=self.converter)
        return eqn

    def benchmark(self, data, sympify_expr=True):
        results = []
        for X,y in data:
            program = self.predict_single(X, y, sympify_expr)
            results.append(program)

        return results

