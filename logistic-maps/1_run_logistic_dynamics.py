"""Run dynamics of systems of coupled logistic maps."""
import os
from itertools import product
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from logistic import LogisticMaps


# Set seed
np.random.seed(303)
# Globals
N_JOBS = 4
DATA_PATH = os.path.realpath(os.path.join('.', 'data'))
os.makedirs(DATA_PATH, exist_ok=True)

# Simulation parameters
R = (3.6, 3.65, 3.7, 3.75, 3.8, 3.85, 3.9, 3.95)
A = (.05, .1, .2, .4, .6, .8)
N = (1, 2, 4, 8, 16, 32, 64)
N_STEPS = (1000,)

# Simulation function
def simulate(idx, params):
    r, alpha, n, n_steps = params
    X = np.random.uniform(0, 1, (n, ))
    Adj = np.ones((n, n), dtype=int)
    # Adj = np.random.randint(0, 2, (n, n))
    lm = LogisticMaps(X, Adj, r=r, alpha=alpha)
    lm.run(n=n_steps, save=True)
    return pd.DataFrame({
        'idx': idx,
        'n': n,
        'r': r,
        'alpha': alpha,
        'n_steps': n_steps,
        'x_i': lm.dynamics[0, :],
        'order': lm.order,
        'var': lm.var
    })

# Run simulations
loop = tuple(enumerate(product(R, A, N, N_STEPS), 1))
results = Parallel(n_jobs=N_JOBS)(
    delayed(simulate)(idx, params) for idx, params in tqdm(loop)
)

df = pd.concat(results)
df.to_csv(os.path.join(DATA_PATH, 'logistic.tsv'), sep="\t", index=False)
