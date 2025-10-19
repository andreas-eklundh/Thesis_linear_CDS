import sympy as sp
import itertools
from scipy.linalg import expm 
import numpy as np

class MultivariatePolynomialGenerator:
    def __init__(self, drift, diffusion, vars, max_degree=3):
        """
        drift: list of sympy expressions [a1, ..., am]
        diffusion: list of lists, matrix [[b11,...,b1r],...,[bm1,...,bmr]]
        vars: list of sympy symbols [z1, ..., zm]
        """
        self.z = vars
        self.a = drift
        self.B = sp.Matrix(diffusion)
        self.var_dim = len(vars)
        self.max_degree = max_degree
        self.Q = self.B * self.B.T   # diffusion covariance
        
        self.basis = self._monomial_basis()
        # store a matrix to do calculations.
        self.basis_mat = sp.Matrix(self.basis)

        
    def _monomial_basis(self):
        """Generate all monomials up to total degree max_degree."""
        basis = []
        for deg in range(self.max_degree + 1):
            for exps in itertools.combinations_with_replacement(range(self.var_dim), deg):
                mon = 1
                powers = [0]*self.var_dim
                for i in exps:
                    powers[i] += 1
                for i, e in enumerate(powers):
                    mon *= self.z[i]**e
                basis.append(mon)
        return basis

    def L(self, f):
        """Generator operator Lf"""
        grad = sp.Matrix([sp.diff(f, zi) for zi in self.z])
        hess = sp.hessian(f, self.z)
        return (sp.Matrix(self.a).dot(grad)
                + 0.5 * sp.trace(self.Q * hess))

    def generator_matrix(self):
        """Compute G such that L(H) = H * G"""
        N = len(self.basis)
        G = sp.zeros(N)
        for j, f in enumerate(self.basis):
            Lf = sp.expand(self.L(f))
            poly = sp.Poly(Lf, *self.z)
            for monom, c in poly.terms():
                # Find the matching basis index
                mon_expr = sp.Mul(*[z**p for z, p in zip(self.z, monom)])
                i = self.basis.index(mon_expr)
                G[i, j] = c
        return G


    # Actually calculate. remainder has just been logic
    def calculate_expected(self,z):
        E_val = np.zeros(self.var_dim)
        z_basis = self.basis_mat.subs(z).evalf()
        G_n = self.generator_matrix()
        G_n_exp = expm(G_n)
        # Loop over indixes
        for i in range(1,(self.var_dim)+1):
            e_i = np.zeros(shape = z_basis.shape[0])
            e_i[i] = 1
            E_val[i-1] = z_basis.T @ G_n_exp @ e_i
        return E_val
    
    def calculate_cov(self,z):
        # Contstruct cov matrix by outer prod of var 
        # NOTE. this is only first moments..
        second_moments = np.zeros(shape = (self.var_dim,self.var_dim))
        
        # Idx to calculate.
        # Get basis and generator.
        z_basis = self.basis_mat.subs(z).evalf()
        G_n = self.generator_matrix()
        G_n_exp = expm(G_n)
        # Start index after first moments.
        idx_start = self.var_dim+1
        for i in range(0,self.var_dim):
            for j in range(i,self.var_dim):
                e_i = np.zeros(shape = z_basis.shape[0])
                e_i[idx_start] = 1
                second_moments[i,j] =  z_basis.T @ G_n_exp @ e_i
                second_moments[j,i] = second_moments[i,j] 
                # increment index
                idx_start += 1

        # Get expected values.
        E_val = self.calculate_expected(z)
        
        E_square = np.outer(E_val, E_val)

        # Get covariance.
        cov = second_moments - E_square

        return cov