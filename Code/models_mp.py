# -*- coding: utf-8 -*-
import numpy as np
import cvxpy as cp
from abc import ABC, abstractmethod
from sklearn_lvq import GlvqModel, GmlvqModel


class ConvexProgram(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def _build_constraints(self, var_x, y, x_orig, corr):
        raise NotImplementedError()

    def _solve(self, prob):
        prob.solve(solver=cp.MOSEK, verbose=False)

    def build_solve_opt(self, x_orig, y, corr, mad=None):
        dim = corr.shape[1]
        
        # Variables
        x = cp.Variable(dim)
        beta = cp.Variable(dim)
        
        # Constants
        c = np.ones(dim)
        z = np.zeros(dim)
        I = np.eye(dim)

        # Construct constraints
        constraints = self._build_constraints(x, y, x_orig, corr)

        # If necessary, construct the weight matrix for the weighted Manhattan distance
        Upsilon = None
        if mad is not None:
            alpha = 1. / mad
            Upsilon = np.diag(alpha)

        # Build the final program
        f = None
        if mad is not None:
            f = cp.Minimize(cp.norm1(x))
        else:
            f = cp.Minimize(cp.norm2(corr @ x))  # Minimize L2 distance

        prob = cp.Problem(f, constraints)
        
        # Solve it!
        self._solve(prob)
        
        x_sol = x.value # Compute final causal counterfactual
        return x_orig + corr @ x_sol, x_sol


class SDP(ABC):
    def __init__(self, **kwds):
        self.epsilon = 1e-2
        self.solver = cp.MOSEK
        self.solver_verbosity = False

        super().__init__(**kwds)
    
    @abstractmethod
    def _build_constraints(self, var_X, var_x, y):
        raise NotImplementedError()
    
    def _solve(self, prob):
        prob.solve(solver=self.solver, verbose=self.solver_verbosity)

    def build_solve_opt(self, x_orig, y, corr, features_whitelist=None, optimizer_args=None):
        if optimizer_args is not None:
            if "solver_verbosity" in optimizer_args:
                self.solver_verbosity = optimizer_args["solver_verbosity"]

        dim = x_orig.shape[0]

        # Variables
        X = cp.Variable((dim, dim), symmetric=True)
        x = cp.Variable((dim, 1))
        one = np.array([[1]]).reshape(1, 1)
        I = np.eye(dim)

        # Construct constraints
        constraints = self._build_constraints(X, x, y, x_orig, corr)
        constraints += [cp.bmat([[X, x], [x.T, one]]) >> 0]

        # If requested, fix some features
        if features_whitelist is not None:
            A = []
            a = []

            for j in range(dim):
                if j not in features_whitelist:
                    t = np.zeros(dim)
                    t[j] = 1.
                    A.append(t)
                    a.append(x_orig[j])
            
            if len(A) != 0:
                A = np.array(A)
                a = np.array(a)

                constraints += [A @ (x_orig + corr @ x) == a]

        # Build the final program
        f = cp.Minimize(cp.trace(I @ X))
        prob = cp.Problem(f, constraints)
        
        # Solve it!
        self._solve(prob)

        x_sol = x.value.reshape(dim) # Compute final causal counterfactual
        return x_orig + corr @ x_sol, x_sol


class Counterfactual(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def compute_counterfactual(self):
        raise NotImplementedError()


class SeparatingHyperplane(Counterfactual, ConvexProgram):
    def __init__(self, w, b, epsilon=1e-5):
        self.w = w
        self.b = b

        self.epsilon = epsilon

    def _build_constraints(self, var_d, y, x_orig, corr):
        if self.w.shape[0] > 1:     # Multiclass problem!
            constraints = []
            for i in range(self.w.shape[0]):
                if i != y:
                    constraints += [(x_orig + corr @ var_d).T @ (self.w[i,:] - self.w[y,:]) + (self.b[i] - self.b[y]) + self.epsilon <= 0]

            return constraints
        else:   # Binary classification!
            if y == 0:
                return [(x_orig + corr @ var_d).T @ self.w.reshape(-1, 1) + self.b + self.epsilon <= 0]
            else:
                return [(x_orig + corr @ var_d).T @ self.w.reshape(-1, 1) + self.b - self.epsilon >= 0]

    def compute_counterfactual(self, x, y, corr, regularizer="l1"):        
        mad = None
        if regularizer == "l1":
            mad = np.ones(x.shape[0])
        
        return self.build_solve_opt(x, y, corr, mad)


class GMLVQ(Counterfactual):
    def __init__(self, model, epsilon=1e-2):
        self.model = model
        self.dim = model.w_[0].shape[0]
        self.epsilon = epsilon
        self.prototypes = model.w_
        self.labels = model.c_w_

    def _solve(self, prob):
        prob.solve(solver=cp.MOSEK, verbose=False)

    def _build_omega(self):
        if isinstance(self.model, GlvqModel):
            return np.eye(self.dim)
        else:
            return np.dot(self.model.omega_.T, self.model.omega_)

    def _compute_counterfactual_target_prototype(self, x_orig, target_prototype, other_prototypes, y_target, corr, features_whitelist=None, mad=None):
        dim = x_orig.shape[0]

        # Variables
        x = cp.Variable(dim)
        beta = cp.Variable(dim)
        
        # Constants
        c = np.ones(dim)
        z = np.zeros(dim)
        I = np.eye(dim)

        # Construct constraints
        constraints = []

        p_i = target_prototype

        Omega = self._build_omega()

        G = []
        b = np.zeros(len(other_prototypes))
        k = 0
        for k in range(len(other_prototypes)):
            p_j = other_prototypes[k]
            G.append(np.dot(Omega, p_j - p_i))
            b[k] = -0.5 * (np.dot(p_i, np.dot(Omega, p_i)) - np.dot(p_j, np.dot(Omega, p_j))) - self.epsilon
        G = np.array(G)
        #print(G, b)

        # If requested, fix the values of some features/dimensions
        A = None
        a = None
        if features_whitelist is not None:
            A = []
            a = []

            for j in range(dim):
                if j not in features_whitelist:
                    t = np.zeros(dim)
                    t[j] = 1.
                    A.append(t)
                    a.append(x_orig[j])
            A = np.array(A)
            a = np.array(a)

        # If necessary, construct the weight matrix for the weighted Manhattan distance
        Upsilon = None
        if mad is not None:
            alpha = 1. / mad
            Upsilon = np.diag(alpha)

        # Build the final program
        f = None
        if mad is not None:
            f = cp.Minimize(cp.norm1(x))
        else:
            f = cp.Minimize(cp.norm2(corr @ x)) # Minimize L2 distance
        constraints += [G @ (x_orig + corr @ x) <= b]

        if A is not None and a is not None:
            constraints += [A @ x == a]
        
        prob = cp.Problem(f, constraints)
        
        # Solve it!
        self._solve(prob)
        
        x_sol = x.value # Compute final counterfactual
        return x_orig + corr @ x_sol, x_sol

    def compute_counterfactual(self, x_orig, y_target, corr, features_whitelist=None, regularizer="l1"):
        mad = None
        if regularizer == "l1":
            mad = np.ones(x_orig.shape[0])
        
        xcf = None
        delta = None
        xcf_dist = float("inf")

        dist = lambda x: np.linalg.norm(x - x_orig, 2)
        if mad is not None:
            dist = lambda x: np.dot(mad, np.abs(x - x_orig))
        
        # Search for suitable prototypes
        target_prototypes = []
        other_prototypes = []
        for p, l in zip(self.prototypes, self.labels):
            if l == y_target:
                target_prototypes.append(p)
            else:
                other_prototypes.append(p)
        
        # Compute a counterfactual for each prototype
        for i in range(len(target_prototypes)):
            try:
                xcf_, delta_ = self._compute_counterfactual_target_prototype(x_orig, target_prototypes[i], other_prototypes, y_target, corr, features_whitelist, mad)
                xcf_proj = xcf_
                ycf_ = self.model.predict([xcf_proj])[0]

                idx = np.argmin([np.linalg.norm(xcf_proj - self.model.w_[j], 2) for j in range(self.model.w_.shape[0])])
                ycf__ = self.model.c_w_[idx]

                if ycf_ == y_target:
                    if dist(xcf_) < xcf_dist:
                        delta = delta_
                        xcf = xcf_
                        xcf_dist = dist(xcf_)
                else:
                    print(ycf_, ycf__, y_target)
            except Exception as ex:
                print(ex)
        
        return xcf, delta