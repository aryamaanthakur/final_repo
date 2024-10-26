import math
from sklearn.metrics import r2_score

class CPScore:
    def __init__(self, ):
        pass

    def __call__(self, y_true, y_pred, N, N0):
        return self.complexity_score(N, N0) - self.performance_score(y_true, y_pred)

    def performance_score(y_true, y_pred):
        return r2_score(y_true, y_pred)

    def complexity_score(N, N0):
        return max(0, math.tanh(N/N0 - 1))