import os
import pickle
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional
from sympy import sympify, sin, cos
from feat import FeatRegressor as FR

@dataclass
class FeatConfig:
    pop_size: Optional[int] = field(default=100)
    gens: Optional[int] = field(default=100)
    max_time: Optional[int] = field(default=2*60)
    max_depth: Optional[int] = field(default=6)
    verbosity: Optional[int] = field(default=2)
    batch_size: Optional[int] = field(default=100)
    functions: Optional[list] = field(default_factory=lambda: ['+','-','*','/','^2','^3','sqrt','sin','cos','exp','log',])
    otype: Optional[int] = field(default='f')

    @classmethod
    def from_yaml(cls, yaml_string, from_file=True):
        if from_file:
            with open(yaml_string) as f:
                yaml_string = f.read()
        
        args_dict = yaml.safe_load(yaml_string)
        try:
            config = FeatConfig(**args_dict)
        except TypeError:
            ExtendedFeatConfig = dataclass(type("ExtendedFeatConfig", (object,), {"__annotations__": args_dict}))
            config = ExtendedFeatConfig(**args_dict)
        return config

class FeatRegressor:
    def __init__(self, config):
        self.config = config
        #self.model = self.get_model()
        self.converter = {'sub': lambda x, y : x - y,
                          'div': lambda x, y : x/y,
                          'mul': lambda x, y : x*y,
                          'add': lambda x, y : x + y,
                          'neg': lambda x    : -x,
                          'pow': lambda x, y : x**y,
                          'sin': lambda x    : sin(x),
                          'cos': lambda x    : cos(x),
                          'inv': lambda x: 1/x,
                          'sqrt': lambda x: x**0.5,
                          'pow3': lambda x: x**3,
                          }

    def get_model(self):
        model = FR(**asdict(self.config))
        
        return model
    
    def predict_single(self, X, y, sympify_expr=True):
        self.model = self.get_model()
        self.model.fit(X, y)

        model_str = self.model.cfeat_.get_eqn()
        model_str = model_str.replace('|','')
        model_str = model_str.replace('^','**')

        if sympify_expr:
            return sympify(model_str, locals=self.converter)
        return model_str

    def benchmark(self, data, sympify_expr=True):
        results = []
        for X,y in data:
            program = self.predict_single(X, y, sympify_expr)
            results.append(program)

        return results

