import numpy as np
from scipy.optimize import minimize, approx_fprime, check_grad
from utils import get_hessian_matrix, load_data, softmax


# N: num observations
# K: num choices
# M: num features
# X[N,K,M]
# Y[N,]
# beta[M,]


def logit(X, beta):
    # x:[N, K, M]
    # beta: [M,]
    u = np.einsum("nkm,m->nk", X, beta)
    return softmax(u, axis=1)


def cross_entropy_loss(beta, x, y, cache):
    # log likelihood + softmax for optimization,
    # y: [N,k], x:[N,K,M], beta:[M,1]
    u = np.einsum("nkm,m->nk", x, beta.reshape(-1))  # [N, K]
    M = np.max(u, axis=-1, keepdims=True)  # [N, 1]
    xbeta = u - M
    exp_xbeta = np.sum(np.exp(xbeta), axis=1, keepdims=True)  # [N, 1]
    left = np.take_along_axis(xbeta, y[:, None], axis=1)
    ll = left - np.log(exp_xbeta)
    cache['beta'] = beta
    return -ll.sum()


def logit_gradient(beta, x, y, cache: dict):
    # y:[N, ], x:[N,K,M], beta:[M, ]
    # beta = cache['beta']
    # assert np.array_equal(beta, cache['beta']) #  scipy package change beta
    y_hat = logit(X=x, beta=beta)
    yx = np.take_along_axis(x, np.expand_dims(y, axis=(1, 2)), axis=1).squeeze(axis=1)  # [N, M]
    px = np.einsum("nk,nkj->nj", y_hat, x)
    gradi = yx - px
    cache['y_hat'] = y_hat
    return -gradi.sum(0)


def logit_hessian(beta, x, y, cache: dict, *args):
    # x:[N,K,M], y_hat:[N,K]
    # assert np.array_equal(beta, cache['beta']) #  scipy package change beta
    # y_hat = cache['y_hat']
    if np.array_equal(beta, cache['beta']):
        y_hat = cache['y_hat']
    else:

        y_hat = logit(X=x, beta=beta)
    px = np.einsum("ik,ikm->im", y_hat, x)  # [N,M]
    xpx = np.subtract(x, np.expand_dims(px, axis=1))  # [N,K,M]
    pxpx = np.einsum("nk,nkm->nkm", y_hat, xpx)
    # pxpx = y_hat[:, :, np.newaxis] * xpx
    hessian = np.einsum("mki,ikj->mj", pxpx.T, x)
    return -hessian


# +++++++++++++++++++++++++++++++++++++
#           Prepare Data      
# -------------------------------------

X, Y = load_data()
# +++++++++++++++++++++++++++++++++++++
#           multinomial logit
# -------------------------------------
cache = {}
rng = np.random.default_rng(seed=3)
result = minimize(fun=cross_entropy_loss,
                  x0=rng.random(size=(2,)), args=(X, Y, cache),
                  method='Newton-CG',
                  jac=logit_gradient,
                  hess=logit_hessian)
beta = result.x
y_hat = logit(X, result.x)  # [N, 2]
hessian = logit_hessian(beta=beta, x=X, y=Y, cache=cache)
omega = np.linalg.inv(-hessian)
se = np.sqrt(np.diag(omega))
gradient = logit_gradient(beta=result.x, y=Y, x=X, cache=cache)
t_value = beta / se
log_loss = cross_entropy_loss(beta=beta, y=Y, x=X, cache=cache)
ll = 2 * log_loss
num_param = len(beta)
AIC = 2 * (log_loss + num_param)
BIC = 2 * log_loss + num_param * np.log(X.shape[0])

# check gradient and hessian function
eps = 1.4901161193847656e-08
approx_fprime(np.array([0.1, 0.2]), cross_entropy_loss, eps, X, Y, cache)
logit_gradient(beta=np.array([0.1, 0.2]), x=X, y=Y, cache=cache)

hessian_matrix = get_hessian_matrix(loss_fn=cross_entropy_loss, beta=beta, y=Y, x=X, tol=np.sqrt(eps), cache=cache)
np.sqrt(1 / np.linalg.svd(hessian_matrix, compute_uv=False))
