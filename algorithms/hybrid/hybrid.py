import numpy as np
from utils.sympy_prefix import prefix_to_sympy
from utils.sympy_deap import sympy_to_deap, deap_to_sympy
from algorithms.hybrid.custom_gp import CustomGP, CustomGPConfig
from algorithms.xval_transformers.dataset import NUM_IDX, PAD_IDX
import random
import sympy as sp

class HybridPredictor:
    def __init__(self, config):
        self.config = config
        self.predictor = self.get_predictor()

    def get_predictor(self):
        if self.config.xval:
            from algorithms.xval_transformers import BeamPredictor
        else:
            from algorithms.transformers import BeamPredictor
        predictor = BeamPredictor(self.config)
        return predictor
    
    def generate_expressions(self, x, num_array=None):
        if num_array is not None:
            candidates = self.predictor.predict(
                x = x.unsqueeze(0),
                num_array = num_array,
                beam_size=self.config.beam_size,
                num_equations=self.config.num_equations
            )
        else:
            candidates = self.predictor.predict(
                x = x.unsqueeze(0),
                beam_size=self.config.beam_size,
                num_equations=self.config.num_equations
            )
        return candidates

    def validate_expressions(self, expressions, num_vars):
        valid = []

        # check for integers too
        for expression in expressions:
            try:
                expression = prefix_to_sympy(expressions)
                if expression not in valid:
                    valid.append(expression)
            except:
                pass
                
        for expression in valid:
            vars = expression.free_symbols
            for var in vars:
                if int(str(var).split("_")[1]) > num_vars:
                    expression.replace(var, sp.Symbol(f"s_{random.randint(0, num_vars)}"))

        return valid

    def chunkify_with_back_sampling(self, X, y):
        N = X.shape[0]
        num_full_chunks = N // self.config.chunk_size
        chunks_X, chunks_y = [], []
        
        # Create full chunks
        for i in range(num_full_chunks):
            chunks_X.append(X[i * self.config.chunk_size:(i + 1) * self.config.chunk_size])
            chunks_y.append(y[i * self.config.chunk_size:(i + 1) * self.config.chunk_size])
        
        # If there are remaining samples, create the last chunk
        remaining_count = N % self.config.chunk_size
        if remaining_count > 0:
            # Gather all previous samples and randomly select from them for the last chunk
            all_previous_indices = np.arange(num_full_chunks * self.config.chunk_size)
            sampled_indices = np.random.choice(all_previous_indices, self.config.chunk_size, replace=False)
            chunks_X.append(X[sampled_indices])
            chunks_y.append(y[sampled_indices])

        return chunks_X, chunks_y

    def get_candidate_equations(self, X, y):
        chunks_X, chunks_y = self.chunkify_with_back_sampling(X, y)
        
        candidate_equations = []
        for Xi, yi in zip(chunks_X, chunks_y):
            candidate_equations.extend(
                self.get_candidate_equations_single(Xi, yi)
            )

        return candidate_equations

    def get_candidate_equations_single(self, X, y):
        x, num_array = self.format_data_for_transformer(X, y)
        expressions = self.generate_expressions(x, num_array)
        expressions = self.validate_expressions(expressions)
        candidates = []
        for expression in expressions:
            candidates.append(sympy_to_deap(expression))
        return candidates
    
    def format_data_for_transformer(self, X, y):
        if self.xval:
            N, n = X.shape
            x = np.concatenate([y, X], axis=1)
            padding = np.ones((N, self.config.max_input_points - n - 1))
            num_array = np.concatenate((x, padding), axis=1)
            x = np.concatenate([np.ones((N, n))*NUM_IDX, padding*PAD_IDX], axis=1)

            return x, num_array
        else:
            print("Coming in future")
            return NotImplementedError
        
    
    def get_gp_predictor(self, num_vars):
        gp_config = CustomGPConfig(
            pop_size=self.config.pop_size,
            cxpb = self.config.cxpb,
            mutpb = self.config.mutpb,
            num_generations = self.config.num_generations,
            num_vars = num_vars
        )
        gp = CustomGP(gp_config)
        return gp
    
    def predict_equation(self, X, y):
        candidates = self.get_candidate_equations(X, y)
        gp = self.get_gp_predictor(X.shape[1])
        points = [(xi.tolist(), yi) for (xi, yi) in zip(X,y)]
        hof = gp(points, candidates)
        return deap_to_sympy(hof[0])
