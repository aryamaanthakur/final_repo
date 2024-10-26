import math
import numpy as np
import operator
from deap import base, creator, tools, gp, algorithms
from dataclasses import dataclass, field, asdict
from typing import Optional

UPPER_LIMIT = 1e7
def protected_div(x1, x2):
    try:
        return x1 / x2
    except ZeroDivisionError:
        return UPPER_LIMIT

def protected_exp(x1):
    try:
        return math.exp(x1) if x1 < 100 else 0.0
    except OverflowError:
        return UPPER_LIMIT

def protected_log(x1):
    try:
        return math.log(x1)
    except:
        return -UPPER_LIMIT

def protected_sqrt(x1):
    try:
        return math.sqrt(x1)
    except:
        return 0

def protected_pow(x1,x2):
    try:
        a = math.pow(x1,x2)
        return a
    except:
        return 1e7

@dataclass
class CustomGPConfig:
    pop_size: Optional[int] = field(default=500)
    cxpb: Optional[float] = field(default=0.7)
    mutpb: Optional[float] = field(default=0.1)
    num_generations: Optional[int] = field(default=15)
    num_vars: Optional[int] = field(default=5)

class CustomGP:
    def __init__(self, config):
        self.config = config
        self.pop_size = config.pop_size
        self.pset = self.get_pset(config.num_vars)
        self.cxpb = self.config.cxpb
        self.mutpb = self.config.mutpb
        self.num_generations = self.config.num_generations

    def get_pset(self, num_vars):
        pset = gp.PrimitiveSet("MAIN", num_vars)
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(math.sin, 1)
        pset.addPrimitive(math.cos, 1)
        pset.addPrimitive(math.tan, 1)
        pset.addPrimitive(math.tanh, 1)
        for i in range(1, 5):
            pset.addTerminal(int(i))
        pset.addPrimitive(protected_div, 2)
        pset.addPrimitive(protected_pow, 2)
        pset.addPrimitive(protected_exp, 1)
        pset.addPrimitive(protected_log, 1)
        pset.addPrimitive(protected_sqrt, 1)
        pset.addTerminal(math.pi, name="pi")
        pset.addTerminal(math.e, name="E")
        rename_kwargs = {"ARG{}".format(i): f"s_{i+1}" for i in range(0, num_vars)}
        pset.renameArguments(**rename_kwargs)
        return pset


    def get_toolbox(self, points):
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=4)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", self.get_initial_population, toolbox=toolbox)
        toolbox.register("compile", gp.compile, pset=self.pset)
        toolbox.register("evaluate", self.evalSymbReg, points=points, pset=self.pset)
        toolbox.register("select", tools.selAutomaticEpsilonLexicase)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=self.pset)
        return toolbox

    @staticmethod
    def get_initial_population(pop_size, seed_exprs, pset, toolbox):
        population = []
        
        for expr in seed_exprs:
            try:
                candidate = creator.Individual.from_string(expr, pset)
                population.append(candidate)
            except:
                continue
        
        print(f"Total seed expressions: {len(seed_exprs)}, Valid expressions used: {len(population)}")
        
        for i in range(pop_size - len(population)):
            random_candidate = toolbox.individual()
            population.append(random_candidate)
        print(population)
        return population

    @staticmethod
    def evalSymbReg(individual, points, pset):
        func = gp.compile(expr=individual, pset=pset)
        sqerrors = (((func(*x) - y)**2)/len(points) for x, y in points)
        return math.fsum(sqerrors),

    def register_stats(self):
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        return stats
    
    def __call__(self, points, candidate_equations):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        toolbox = self.get_toolbox(points)
        population = toolbox.population(self.pop_size, candidate_equations, self.pset)

        fitness_results = [self.evalSymbReg(individual, points, self.pset) for individual in population]

        for individual, fitness_score in zip(population, fitness_results):
            individual.fitness.values = fitness_score

        hof = tools.HallOfFame(5)
        stats = self.register_stats()
        population, log = algorithms.eaSimple(
            population,
            toolbox,
            self.cxpb,
            self.mutpb,
            self.num_generations,
            stats=stats,
            halloffame=hof,
            verbose=True
        )

        TSS = 0.0
        mean_y = sum(y for _, y in points) / len(points)
        for _, y in points:
            TSS += (y - mean_y) ** 2
        r2_score = 1 - (float(hof[0].fitness.values[0]) / TSS)
        print("R2_score:", r2_score)

        print("Best individual:", hof[0])
        print("Fitness:", hof[0].fitness.values)
        return hof, r2_score