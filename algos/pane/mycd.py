""" Non-negative matrix factorization
"""
# Author: Vlad Niculae
#         Lars Buitinck
#         Mathieu Blondel <mathieu@mblondel.org>
#         Tom Dupre la Tour
# License: BSD 3 clause

import numbers
import numpy as np
import scipy.sparse as sp
import time
import warnings
from math import sqrt

import pyximport
pyximport.install()

import cdnmf_fast
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from sklearn.utils.validation import check_is_fitted, check_non_negative

EPSILON = np.finfo(np.float32).eps


def norm(x):
    return sqrt(squared_norm(x))


def trace_dot(X, Y):
    return np.dot(X.ravel(), Y.ravel())


def _check_init(A, shape, whom):
    A = check_array(A)
    if np.shape(A) != shape:
        raise ValueError('Array with wrong shape passed to %s. Expected %s, '
                         'but got %s ' % (whom, shape, np.shape(A)))
    if np.max(A) == 0:
        raise ValueError('Array passed to %s is full of zeros.' % whom)

def _beta_divergence(X, W, H, beta, square_root=False):
    beta = _beta_loss_to_float(beta)

    # The method can be called with scalars
    if not sp.issparse(X):
        X = np.atleast_2d(X)
    W = np.atleast_2d(W)
    H = np.atleast_2d(H)

    # Frobenius norm
    if beta == 2:
        # Avoid the creation of the dense np.dot(W, H) if X is sparse.
        if sp.issparse(X):
            norm_X = np.dot(X.data, X.data)
            norm_WH = trace_dot(np.dot(np.dot(W.T, W), H), H)
            cross_prod = trace_dot((X * H.T), W)
            res = (norm_X + norm_WH - 2. * cross_prod) / 2.
        else:
            res = squared_norm(X - np.dot(W, H)) / 2.

        if square_root:
            return np.sqrt(res * 2)
        else:
            return res

    if sp.issparse(X):
        # compute np.dot(W, H) only where X is nonzero
        WH_data = _special_sparse_dot(W, H, X).data
        X_data = X.data
    else:
        WH = np.dot(W, H)
        WH_data = WH.ravel()
        X_data = X.ravel()

    # do not affect the zeros: here 0 ** (-1) = 0 and not infinity
    indices = X_data > EPSILON
    WH_data = WH_data[indices]
    X_data = X_data[indices]

    # used to avoid division by zero
    WH_data[WH_data == 0] = EPSILON

    # generalized Kullback-Leibler divergence
    if beta == 1:
        # fast and memory efficient computation of np.sum(np.dot(W, H))
        sum_WH = np.dot(np.sum(W, axis=0), np.sum(H, axis=1))
        # computes np.sum(X * log(X / WH)) only where X is nonzero
        div = X_data / WH_data
        res = np.dot(X_data, np.log(div))
        # add full np.sum(np.dot(W, H)) - np.sum(X)
        res += sum_WH - X_data.sum()

    # Itakura-Saito divergence
    elif beta == 0:
        div = X_data / WH_data
        res = np.sum(div) - np.product(X.shape) - np.sum(np.log(div))

    # beta-divergence, beta not in (0, 1, 2)
    else:
        if sp.issparse(X):
            # slow loop, but memory efficient computation of :
            # np.sum(np.dot(W, H) ** beta)
            sum_WH_beta = 0
            for i in range(X.shape[1]):
                sum_WH_beta += np.sum(np.dot(W, H[:, i]) ** beta)

        else:
            sum_WH_beta = np.sum(WH ** beta)

        sum_X_WH = np.dot(X_data, WH_data ** (beta - 1))
        res = (X_data ** beta).sum() - beta * sum_X_WH
        res += sum_WH_beta * (beta - 1)
        res /= beta * (beta - 1)

    if square_root:
        return np.sqrt(2 * res)
    else:
        return res


def _special_sparse_dot(W, H, X):
    """Computes np.dot(W, H), only where X is non zero."""
    if sp.issparse(X):
        ii, jj = X.nonzero()
        n_vals = ii.shape[0]
        dot_vals = np.empty(n_vals)
        n_components = W.shape[1]

        batch_size = max(n_components, n_vals // n_components)
        for start in range(0, n_vals, batch_size):
            batch = slice(start, start + batch_size)
            dot_vals[batch] = np.multiply(W[ii[batch], :],
                                          H.T[jj[batch], :]).sum(axis=1)

        WH = sp.coo_matrix((dot_vals, (ii, jj)), shape=X.shape)
        return WH.tocsr()
    else:
        return np.dot(W, H)


def _check_string_param(regularization, beta_loss):
    allowed_regularization = ('both', 'components', 'transformation', None)
    if regularization not in allowed_regularization:
        raise ValueError(
            'Invalid regularization parameter: got %r instead of one of %r' %
            (regularization, allowed_regularization))

    beta_loss = _beta_loss_to_float(beta_loss)
    return beta_loss


def _beta_loss_to_float(beta_loss):
    """Convert string beta_loss to float"""
    allowed_beta_loss = {'frobenius': 2,
                         'kullback-leibler': 1,
                         'itakura-saito': 0}
    if isinstance(beta_loss, str) and beta_loss in allowed_beta_loss:
        beta_loss = allowed_beta_loss[beta_loss]

    if not isinstance(beta_loss, numbers.Number):
        raise ValueError('Invalid beta_loss parameter: got %r instead '
                         'of one of %r, or a float.' %
                         (beta_loss, allowed_beta_loss.keys()))
    return beta_loss


def _update_coordinate_descent(X, W, Ht, shuffle, random_state):
    n_components = Ht.shape[1]

    HHt = np.dot(Ht.T, Ht)
    XHt = safe_sparse_dot(X, Ht)

    if shuffle:
        permutation = random_state.permutation(n_components)
    else:
        permutation = np.arange(n_components)
    # The following seems to be required on 64-bit Windows w/ Python 3.5.
    permutation = np.asarray(permutation, dtype=np.intp)
    return cdnmf_fast._update_cdnmf_fast2(W, HHt, XHt, permutation)



def _fit_coordinate_descent(X, W, H, tol=1e-4, max_iter=200, update_H=True,
                            verbose=0, shuffle=False, random_state=None):
    # so W and Ht are both in C order in memory
    Ht = check_array(H.T, order='C')
    X = check_array(X, accept_sparse='csr')

    rng = check_random_state(random_state)

    for n_iter in range(max_iter):
        #print("iter: %d/%d"%(n_iter,max_iter))
        violation = 0.

        # Update W
        violation += _update_coordinate_descent(X, W, Ht, shuffle, rng)
        # Update H
        if update_H:
            violation += _update_coordinate_descent(X.T, Ht, W, shuffle, rng)

        if n_iter == 0:
            violation_init = violation

        if violation_init == 0:
            break

        if verbose:
            print("violation:", violation / violation_init)

        if violation / violation_init <= tol:
            if verbose:
                print("Converged at iteration", n_iter + 1)
            break

    return W, Ht.T, n_iter


def non_negative_factorization(X, W, H, n_components=None, update_H=True, 
                               beta_loss='frobenius', tol=1e-4,
                               max_iter=200, alpha=0., l1_ratio=0.,
                               regularization=None, random_state=None,
                               verbose=0, shuffle=False):

    X = check_array(X, accept_sparse=('csr', 'csc'), dtype=float)
    check_non_negative(X, "NMF (input X)")
    beta_loss = _check_string_param(regularization, beta_loss)

    if X.min() == 0 and beta_loss <= 0:
        raise ValueError("When beta_loss <= 0 and X contains zeros, "
                         "the solver may diverge. Please add small values to "
                         "X, or use a positive beta_loss.")

    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features

    if not isinstance(n_components, numbers.Integral) or n_components <= 0:
        raise ValueError("Number of components must be a positive integer;"
                         " got (n_components=%r)" % n_components)
    if not isinstance(max_iter, numbers.Integral) or max_iter < 0:
        raise ValueError("Maximum number of iterations must be a positive "
                         "integer; got (max_iter=%r)" % max_iter)
    if not isinstance(tol, numbers.Number) or tol < 0:
        raise ValueError("Tolerance for stopping criteria must be "
                         "positive; got (tol=%r)" % tol)

    if H is not None and W is not None:
        _check_init(H, (n_components, n_features), "NMF (input H)")
        _check_init(W, (n_samples, n_components), "NMF (input W)")
    elif H is not None and W is None:
        _check_init(H, (n_components, n_features), "NMF (input H)")
        if H.dtype != X.dtype:
            raise TypeError("H should have the same dtype as X. Got H.dtype = "
                            "{}.".format(H.dtype))
        W = np.zeros((n_samples, n_components), dtype=X.dtype)
    elif H is None and W is None:
        avg = np.sqrt(X.mean() / n_components)
        rng = check_random_state(random_state)
        H = avg * rng.randn(n_components, n_features).astype(X.dtype, copy=False)
        W = avg * rng.randn(n_samples, n_components).astype(X.dtype, copy=False)
        np.abs(H, out=H)
        np.abs(W, out=W)


    W, H, n_iter = _fit_coordinate_descent(X, W, H, tol, max_iter,
                                               update_H=update_H,
                                               verbose=verbose,
                                               shuffle=shuffle,
                                               random_state=random_state)

    if n_iter == max_iter and tol > 0:
        warnings.warn("Maximum number of iteration %d reached. Increase it to"
                      " improve convergence." % max_iter, ConvergenceWarning)

    return W, H, n_iter


class NMF(TransformerMixin, BaseEstimator):
    def __init__(self, n_components=None, updateH=True,
                 beta_loss='frobenius', tol=1e-4, max_iter=200,
                 random_state=None, alpha=0., l1_ratio=0., verbose=0,
                 shuffle=False):
        self.n_components = n_components
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.verbose = verbose
        self.shuffle = shuffle
        self.updateH = updateH

    def _more_tags(self):
        return {'requires_positive_X': True}

    def fit_transform(self, X, W=None, H=None, y=None):
        X = check_array(X, accept_sparse=('csr', 'csc'), dtype=float)

        W, H, n_iter_ = non_negative_factorization(
            X=X, W=W, H=H, n_components=self.n_components, 
            update_H=self.updateH, beta_loss=self.beta_loss,
            tol=self.tol, max_iter=self.max_iter, alpha=self.alpha,
            l1_ratio=self.l1_ratio, regularization='both',
            random_state=self.random_state, verbose=self.verbose,
            shuffle=self.shuffle)

        self.reconstruction_err_ = _beta_divergence(X, W, H, self.beta_loss,
                                                    square_root=True)

        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter_

        return W

    def fit(self, X, y=None, **params):
        self.fit_transform(X, **params)
        return self

    def transform(self, X):
        check_is_fitted(self)

        W, _, n_iter_ = non_negative_factorization(
            X=X, W=None, H=self.components_, n_components=self.n_components_, update_H=False, 
            beta_loss=self.beta_loss, tol=self.tol, max_iter=self.max_iter,
            alpha=self.alpha, l1_ratio=self.l1_ratio, regularization='both',
            random_state=self.random_state, verbose=self.verbose,
            shuffle=self.shuffle)

        return W

    def inverse_transform(self, W):
        check_is_fitted(self)
        return np.dot(W, self.components_)
        