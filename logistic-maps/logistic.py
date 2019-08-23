"""Simulator for systems of coupled logistic maps."""
import numpy as np


def _logistic_step(X, A, r, alpha):
    N = X.shape[0]
    if N == 1:
        return X*r*(1-X)
    return (1-alpha)*(X*r*(1-X)) + (alpha/N)*(A@(X*r*(1-X)))


class LogisticMaps:
    """System of coupled logistic maps.

    Attributes
    ----------
    X : (N, ) array_like[float]
        State vector for logistic maps.
        All values should be between 0 and 1.
    A : (N, N) array_like[int]
        Adjacency matrix defining the topology of the system.
    r : float
        Control parameter for logistic maps.
        Should be between 0 and 4 and greater than ~3.54 for chaos.
    alpha : float
        Coupling parameter. Sets the strength of coupling between
        any two connected logistic maps.
    dynamics : (N, t) array_like[float]
        Full view of the system's dynamics.
        Each column is a state vector at time t.
    order : (t,) array_like[float]
        Vector of order parameter values.
        These are just average states over all logistic maps.
    var : (t,) array_like[float]
        Vector of state variances.
        This is useful for assessing the degree of synchronization.
    """
    def __init__(self, X0, A, r=3.6, alpha=.3):
        """Initialization method."""
        self.X = X0
        self.A = A
        np.fill_diagonal(self.A, 0)
        self.r = r
        self.alpha = alpha
        self._dynamics = []
        self._order = []
        self._var = []

    @property
    def N(self):
        return self.X.shape[0]

    @property
    def dynamics(self):
        return np.column_stack(self._dynamics)

    @property
    def order(self):
        return np.array(self._order)

    @property
    def var(self):
        return np.array(self._var)

    def run(self, n=1, save=True):
        """Run dynamics of coupled logistic maps.

        Parameters
        ----------
        n : int
            Number of time steps to run for.
        save : bool
            Should dynamics be saved.
        """
        def save_step(X):
            if save:
                self._dynamics.append(X.copy())
                self._order.append(X.mean())
                if X.shape[0] > 1:
                    self._var.append(((X - X.mean())**2).sum() / (self.N - 1))
                else:
                    self._var.append(0)

        save_step(self.X)
        for _ in range(n):
            X = _logistic_step(self.X, self.A, self.r, self.alpha)
            save_step(X)
            self.X = X
