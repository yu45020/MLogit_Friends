import numpy as np
from scipy.optimize import minimize, approx_fprime
from utils import softmax, get_hessian_matrix, load_data


# N: num observations
# K: num choices
# M: num features
# Q: num latent class
# L: num features for latent class
# Data:
# X[N,T, K,M]: covoriate for choice
# H[N,L]: covoriate for latent class
# Y[N,T]
# Parameter:
# beta[M, Q]  # main logit parameters
# gamma [L, Q] # latent class parameters


def reshape_parameters(beta, num_x_param, h_param, num_lc):
    # beta: np.array

    first_qm = int(num_lc * num_x_param)
    x_beta = np.reshape(beta[: first_qm], (num_x_param, num_lc))  # [M, Q]

    if num_lc == 1:
        lc_beta = np.zeros((h_param, 1))
    else:
        lc_beta = np.reshape(beta[first_qm:], (-1, num_lc - 1))  # [T, Q]
        # default is the first class
        lc_beta = np.concatenate([np.zeros((lc_beta.shape[0], 1)), lc_beta], axis=-1)

    return x_beta, lc_beta


def create_parameter_names(num_x_param, num_lc_param, num_lc_class, x_beta_name: list, lc_name: list):
    # beta: [M, Q], first row is constant
    # lc_beta: [L, Q]
    # out: list

    L = num_lc_param
    out = []
    for m in range(num_x_param):
        for q in range(num_lc_class):
            out.append(f'{x_beta_name[m]}.LC.{q}')
    if num_lc_class > 1:
        for l in range(L):
            for q in range(1, num_lc_class):
                out.append(f'{lc_name[l]}.LC.{q}')

    return out


def cross_entropy_loss(beta, x, y, h, num_lc, cache: dict):
    # log likelihood of softmax for optimization
    # y: [N,T], x:[N,T,K,M], beta:[M, Q], h:[N,T,L]
    # num_lc: int number of latent class (fixed over time)
    # beta = np.random.rand(num_lc * X.shape[-1] + (num_lc - 1))
    x_beta, gamma = reshape_parameters(beta, x.shape[-1], h.shape[2], num_lc)
    assert h.shape[2] == gamma.shape[0]
    assert x.shape[3] == x_beta.shape[0]
    assert x_beta.shape[1] == gamma.shape[1]
    U_tkq = np.einsum("ntkm,mq->ntkq", x, x_beta)
    P_tkq = softmax(U_tkq, axis=2)  # [N, T, K, Q]

    P_tq = np.take_along_axis(P_tkq, y[:, :, np.newaxis, np.newaxis], axis=2)  # [N, T, 1, Q]
    # equivalent to  P_q = P_tq.prod(axis=(1, 2))
    P_q = np.log(P_tq).sum(axis=(1, 2))
    P_q = np.exp(P_q)  # [N, Q]

    if num_lc == 1:
        Pw_q = np.ones(shape=(h.shape[0], gamma.shape[1]))
    else:
        W_q = np.einsum("ntm,mq->nq", h, gamma)
        Pw_q = softmax(W_q, axis=-1)  # [N,Q]

    L_i = np.einsum("nq,nq->n", P_q, Pw_q)  # [N, K]
    LL = -np.log(L_i).sum()

    cache.update({'x_beta': x_beta, 'gamma': gamma,
                  'L_i': L_i,
                  'P_tkq': P_tkq,  # [N,T,K,Q]
                  'P_q': P_q,  # [N,T,Q]
                  'Pw_q': Pw_q,  # [N,Q]
                  'beta': beta
                  })

    return LL


def logit_latent_gradient(beta, x, y, h, num_lc, cache: dict):
    # x_beta
    # assert np.array_equal(beta, cache['beta']) #  scipy package change beta
    if not np.array_equal(beta, cache['beta']):
        LL = cross_entropy_loss(beta, x, y, h, num_lc, cache)  # update cache
    li2 = np.expand_dims(cache['L_i'], axis=1)  # [N,T(1), K]
    liwq = cache['P_q'] * cache['Pw_q'] / li2
    yxp = np.einsum("ntkm,ntkq->nmq", X, (cache['ytk'] - cache['P_tkq']))
    grad_x_beta = -np.einsum("nq,nmq->mq", liwq, yxp).reshape(-1)

    # gamma
    wll = cache['Pw_q'] * (cache['P_q'] - li2) / li2  # [N,Q]
    grad_gamma = -np.einsum('ntl,nq->lq', H, wll)
    if num_lc == 1:
        grad_gamma = []
    else:
        grad_gamma = grad_gamma[:, 1:].reshape(-1)
    param_grad = np.concatenate([grad_x_beta, grad_gamma])
    return param_grad


# +++++++++++++++++++++++++++++++++++++
#           Prepare Data
# -------------------------------------

X, Y = load_data()
Y = Y.reshape(300, 10)

# convert data to [N, K, M]; M: constant, price, state
X = X.reshape((300, 10, 2, 2))
# assume no covariates to determine latent class
H = np.zeros((300, 10, 1))
H[:, :, 0] = 1  # the first class is default

cache = {}
N = 300
T = 10
K = 2
num_lc = 1  # Q
cache['ytk'] = np.zeros(shape=(N, T, K, num_lc))
np.put_along_axis(cache['ytk'], np.expand_dims(Y, axis=(2, 3)), axis=2, values=1)

rng = np.random.default_rng(seed=2)
num_parameters = num_lc * X.shape[-1] + num_lc - 1
beta = rng.random(size=(num_parameters,))

eps = 1.4901161193847656e-08
# check gradient
approx_grad = approx_fprime(beta, cross_entropy_loss, eps, X, Y, H, num_lc, cache)
analytic_grad = logit_latent_gradient(beta, X, Y, H, num_lc, cache)
approx_grad - analytic_grad

# optim
result = minimize(fun=cross_entropy_loss,
                  x0=beta,
                  args=(X, Y, H, num_lc, cache),
                  method='Newton-CG',
                  # method='CG',  # "Nelder-Mead",  # 'Nelder-Mead',
                  options={'maxiter': 10000},
                  jac=logit_latent_gradient)
beta = result.x
hessian_matrix = get_hessian_matrix(loss_fn=cross_entropy_loss,
                                    beta=beta, y=Y, x=X, h=H, num_lc=num_lc, tol=np.sqrt(eps),
                                    cache=cache)
# beta_sd = np.sqrt(np.diag(np.linalg.pinv(hessian_matrix)))
se = np.sqrt(1 / np.linalg.svd(hessian_matrix, compute_uv=False))
t_value = beta / se
log_loss = result.fun
num_param = len(beta)
AIC = 2 * (log_loss + num_param)
BIC = 2 * log_loss + num_param * np.log(X.shape[0])
# match the same order of beta
create_parameter_names(num_x_param=X.shape[-1], num_lc_param=H.shape[-1],
                       num_lc_class=num_lc,
                       x_beta_name=['const', 'price'],
                       lc_name=['lc_const'])

