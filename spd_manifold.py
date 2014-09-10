"""
Operations on the manifold of SPD matrices and mapping to a flat space.

"""
import numpy as np
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs

def my_stack(arrays):
    return np.concatenate([a[np.newaxis] for a in arrays])

# Bypass scipy for faster eigh (and dangerous: Nan will kill it)
my_eigh, = get_lapack_funcs(('syevr', ), np.zeros(1))

def frobenius(mat):
    """ Return the Frobenius norm
    """
    return np.sqrt((mat**2).sum())/mat.size


def sqrtm(mat):
    """ Matrix square-root, for symetric positive definite matrices.
    """
    vals, vecs, success_flag = my_eigh(mat)
    return np.dot(vecs*np.sqrt(vals), vecs.T)


def inv_sqrtm(mat):
    """ Inverse of matrix square-root, for symetric positive definite matrices.
    """
    vals, vecs, success_flag = my_eigh(mat)
    return np.dot(vecs/np.sqrt(vals), vecs.T)


def expm(mat):
    """ Matrix exponential, for symetric positive definite matrices.
    """
    vals, vecs = linalg.eigh(mat)
    return np.dot(vecs*np.exp(vals), vecs.T)


def logm(mat):
    """ Matrix log, for symetric positive definite matrices.
    """
    vals, vecs, success_flag = my_eigh(mat)
    return np.dot(vecs*np.log(vals), vecs.T)


def log_map(x, displacement, mean=False):
    """ The Riemannian log map at point 'displacement'.
        If several points are given, the mean is returned.

        See algorithm 2 of Fletcher and Joshi, Sig Proc 87 (2007) 250
    """
    #x = np.asanyarray(x)
    vals, vecs, success_flag = my_eigh(displacement)
    sqrt_displacement = np.dot(vecs*np.sqrt(vals), vecs.T)
    whitening = np.dot(vecs/np.sqrt(vals), vecs.T)
    if len(x.shape) == 2:
        log_x = logm(np.dot(np.dot(whitening, x), whitening))
        return np.dot(np.dot(sqrt_displacement, x), sqrt_displacement)
    log_x = my_stack(
                [logm(np.dot(np.dot(whitening, m), whitening)) for m in x])
    if mean:
        x = np.mean(log_x, axis=0)
        return np.dot(np.dot(sqrt_displacement, x), sqrt_displacement)
    return my_stack([np.dot(np.dot(sqrt_displacement, x), sqrt_displacement)
            for x in log_x])
    


def exp_map(x, displacement):
    """ The Riemannian exp map at point 'displacement'.

        See algorithm 1 of Fletcher and Joshi, Sig Proc 87 (2007) 250
    """
    vals, vecs, success_flag = my_eigh(displacement)
    sqrt_displacement = np.dot(vecs*np.sqrt(vals), vecs.T)
    whitening = np.dot(vecs/np.sqrt(vals), vecs.T)
    return np.dot(np.dot(sqrt_displacement,
                    expm(
                        np.dot(np.dot(whitening, x), whitening)
                    )), 
                sqrt_displacement)


def log_mean(population_covs, eps=1e-6):
    """ Find the Riemannien mean of the the covariances.

        See algorithm 3 of Fletcher and Joshi, Sig Proc 87 (2007) 250
    """
    step = 1
    mean = np.mean(population_covs, axis=0)
    direction = log_map(population_covs, mean, mean=True)
    while frobenius(direction) > eps:
        mean = exp_map(step*direction, mean)
        new_direction = log_map(population_covs, mean, mean=True)
        if frobenius(new_direction) < frobenius(direction):
            direction = new_direction
        else:
            step = .5*step
    return mean


def projection(subject_cov, population_covs, whitening=None):
    subject_cov = np.ascontiguousarray(subject_cov)
    if whitening is None:
        whitening = inv_sqrtm(np.mean(population_covs, axis=0))
    if len(subject_cov.shape)==3:
        return my_stack([ np.dot(np.dot(whitening, s), whitening)
                         for s in subject_cov ])
    return np.dot(np.dot(whitening, subject_cov), whitening)


def riemannian_projection(subject_cov, population_covs, whitening=None):
    pop_mean = log_mean(population_covs)
    if len(subject_cov.shape)==3:
        return my_stack(log_map(subject_cov, pop_mean))
    return log_map(subject_cov, pop_mean)


def sym_to_vec(sym):
    sym = np.copy(sym)
    sym -= np.eye(sym.shape[-1])
    # the sqrt(2) factor
    sym *= np.sqrt(2)
    sym += (1 - np.sqrt(2))/np.sqrt(2)*np.diag(np.diag(sym))
    mask = np.tril(np.ones(sym.shape[-2:])).astype(np.bool)
    return sym[..., mask]
 

def vec_to_sym(vec, shape):
    mask = np.tril(np.ones(shape)).astype(np.bool)
    sym = np.zeros(vec.shape[:-1] + mask.shape, vec.dtype)
    sym[..., mask] = vec
    sym -= (1 - np.sqrt(2))*np.diag(np.diag(sym))
    sym /= np.sqrt(2)
    sym += np.tril(sym, k=-1).T
    sym += np.eye(sym.shape[-1])
    return sym
