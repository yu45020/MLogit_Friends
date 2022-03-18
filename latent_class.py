import numpy as np
from scipy.optimize import minimize, approx_fprime
from utils import softmax, get_hessian_matrix, load_data


# N: num observations
# K: num choices
# M: num features
# Q: num latent class
# L: num features for latent class
# X[N,K,M]: covoriate for choice
# H[N,T]: covoriate for latent class
# Y[N,]
# beta[M, Q]  # main logit parameters
# gamma [L,Q] # latent class parameters
# statsmodels.base.model.GenericLikelihoodModel¶


def reshape_parameters(beta, num_x_param, h_param, num_lc):
    # beta: np.array

    first_qm = int(num_lc * num_x_param)
    x_beta = np.reshape(beta[: first_qm], (num_x_param, num_lc))  # [M, Q]

    if num_lc == 1:
        lc_beta = np.ones((h_param, 1))
    else:
        lc_beta = np.reshape(beta[first_qm:], (-1, num_lc - 1))  # [T, Q]
        # default is the first class
        lc_beta = np.concatenate([np.zeros((lc_beta.shape[0], 1)), lc_beta], axis=-1)

    return x_beta, lc_beta


def create_parameter_names(num_x_param, num_lc_param, num_lc_class, x_beta_name: list, lc_name: list):
    # beta: [M, Q], first row is constant
    # lc_beta: [L, Q]
    # out: list

    M, Q = num_x_param, num_lc_class
    L = num_lc_param
    out = []
    for m in range(M):
        for q in range(Q):
            out.append(f'{x_beta_name[m]}.LC.{q}')
    if Q == 1:
        for l in range(L):
            out.append(f'{lc_name[l]}.LC.{1}')
    else:
        for l in range(L):
            for q in range(1, Q):
                out.append(f'{lc_name[l]}.LC.{q}')

    return out


def cross_entropy_loss(beta, y, x, h, num_lc):
    # log likelihood of softmax for optimization
    # y: [N,T], x:[N,T,K,M], beta:[M, Q], h:[N,T,L]
    # num_lc: int number of latent class (fixed over time)
    # beta = np.random.rand(num_lc * X.shape[-1] + (num_lc - 1))
    x_beta, lc_beta = reshape_parameters(beta, x.shape[-1], h.shape[2], num_lc)
    assert h.shape[2] == lc_beta.shape[0]
    assert x.shape[3] == x_beta.shape[0]
    assert x_beta.shape[1] == lc_beta.shape[1]
    u_q = np.einsum("ntkm,mq->ntkq", x, x_beta)
    prob = softmax(u_q, axis=2)  # [N, T, K, Q]

    p_uq = np.take_along_axis(prob, y[:, :, np.newaxis, np.newaxis], axis=2)  # [N, T, 1, Q]
    # p_uq = p_uq.prod(axis=(1, 2))
    p_uq = np.log(p_uq).sum(axis=(1, 2))
    p_uq = np.exp(p_uq)  # [N, Q]

    w_q = np.einsum("ntm,mq->nq", h, lc_beta)
    p_wq = softmax(w_q, axis=-1)  # [N,Q]
    L = np.einsum("nq,nq->n", p_uq, p_wq)  # [N, K]
    LL = -np.log(L).sum()
    return LL, L, prob, p_uq, p_wq


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

beta = np.linspace(0.1, 0.5, num=3)  # np.array([0.3] * 8)
# LL, L, prob, p_uq, p_wq
num_lc = 1
result = cross_entropy_loss(beta, Y, X, H, num_lc)
create_parameter_names(num_x_param=2, num_lc_param=1, num_lc_class=2,
                       x_beta_name=['const', 'price'],
                       lc_name=['lc_const'])
result[0]
li = result[1]  # [N]
x = X  # [N, T, K, M]

y = Y  # [N,T]
h = H  # [N, T, L]
ytk = np.zeros(shape=(300, 10, 2))
x_beta, lc_beta = reshape_parameters(beta, x.shape[-1], h.shape[1], num_lc)

prob = result[2]  # [N, T, K, Q]
prob_uq = result[3]  # [N, Q]
prob_lc = result[4]  # [N, Q]
#
# +++++++++++++++++++++++++++++++++++++
#           Gradient
# -------------------------------------
# beta
li2 = np.expand_dims(li, axis=1)
liwq = (prob_uq * prob_lc) / li2  # [N, Q]

ytk = np.zeros_like(prob)  # [N,T,K,Q]
np.put_along_axis(ytk, np.expand_dims(Y, axis=(2, 3)), axis=2, values=1)
yxp = np.einsum("ntkm,ntkq->nmq", X, (ytk - prob))  # [N,M,Q]
-np.einsum("nq,nmq->mq", liwq, yxp)
beta_grad = -np.einsum("nq,nmq->mq", liwq, yxp).reshape(-1)

#
# xk = np.take_along_axis(X, y[:, :, np.newaxis, np.newaxis], axis=2)  # [N, T, 1, Q]
# pk = np.take_along_axis(prob, y[:, :, np.newaxis, np.newaxis], axis=2)  # [N, T, 1, Q]
# yxp = np.einsum("ntkm,ntkq->nmq", xk, (pk - np.power(pk, 2)))  # (prob - np.power(prob, 2))
# yxp1 = np.einsum("ntkm,ntkq->nmq", xk, pk)
# yxp2 = np.einsum("ntkm,ntkq->nmq", xk, np.power(pk, 2))
# yxp = yxp1-yxp2
# yxp = np.take_along_axis(yxp, np.expand_dims(y, axis=(2, 3, 4)), axis=2)  # ntkmq
# yxp = yxp.sum(axis=(1, 2))
# -np.einsum("nq,nmq->mq", liwq, yxp)
# 1.08203125,   47.59963989,   39.47052002,
# gamma # 8.2288208
wll = prob_lc * (prob_uq - li2) / li2  # [N,Q]
-np.einsum('ntl,nq->lq', H, wll)[:, 1:]
gamma_grad = -np.einsum('ntl,nq->lq', H, wll)[:, 1:].reshape(-1)
param_grad = np.concatenate([beta_grad, gamma_grad])
approx_hes = np.einsum("n,j->nj", param_grad, param_grad)


# np.sqrt(np.diag(np.linalg.pinv(approx_hes)))


def loss_fn(beta, y, x, h, num_lc):
    # log likelihood of softmax for optimization
    # y: [N,T], x:[N,T,K,M], beta:[M, Q], h:[N,T]
    # num_lc: int number of latent class (fixed over time)
    # beta = np.random.rand(num_lc * X.shape[-1] + (num_lc - 1))
    x_beta, lc_beta = reshape_parameters(beta, x.shape[-1], h.shape[1], num_lc)

    u_q = np.einsum("ntkm,mq->ntkq", x, x_beta)
    prob = softmax(u_q, axis=2)  # [N, T, K, Q]

    p_uq = np.take_along_axis(prob, y[:, :, np.newaxis, np.newaxis], axis=2)  # [N, T, 1, Q]

    p_uq = np.log(p_uq).sum(axis=(1, 2))
    p_uq = np.exp(p_uq)  # [N, Q]

    w_q = np.einsum("ntm,mq->nq", h, lc_beta)
    p_wq = softmax(w_q, axis=-1)  # [N,Q]
    L = np.einsum("nq,nq->n", p_uq, p_wq)  # [N, K]
    LL = -np.log(L).sum()
    return LL


approx_fprime(beta, loss_fn, 1.4901161193847656e-08, y, x, h, num_lc)

hessian_matrix = get_hessian_matrix(loss_fn=loss_fn, beta=beta, y=Y, x=X, h=H, num_lc=num_lc,
                                    tol=1.4901161193847656e-4)
beta_sd = np.sqrt(np.diag(np.linalg.pinv(hessian_matrix)))

a = np.linalg.svd(-hessian_matrix, compute_uv=False)
1 / a


# Trash
# +++++++++++++++++++++++++++++++++++++
#
# -------------------------------------


# wliq = prob_lc * prob_uq / np.expand_dims(li, axis=1)  # [N,Q]
# xk = np.take_along_axis(x, np.expand_dims(y, axis=(2, 3)), axis=2)
# prob_k = np.take_along_axis(prob, np.expand_dims(y, axis=(2, 3)), axis=2)
# xpp = np.einsum("ntkm,ntkq->nmq", xk, (1 - prob_k))
#
# np.einsum("nmq,nq->mq", xpp, wliq)
#
# # ypp = np.einsum("ntk,ntk->ntk", ytk, np.power(prob[:, :, :, 0], ytk - 1))
# # xpp = np.einsum("ntk,ntkm->ntkm", ypp, xpp)
# # xpp = np.take_along_axis(xpp, np.expand_dims(y, axis=(2, 3)), axis=2)  # [N,T,1,M]
# xpp = xpp.prod(axis=(1, 2))  # [N,M]
# wxpp = np.einsum("n,nm->nm", prob_lc[:, 0], xpp)
# np.sum(wxpp / np.expand_dims(li, axis=1), axis=0)
# # gradient for beta
# p_uq = np.take_along_axis(prob, y[:, :, np.newaxis, np.newaxis], axis=2)  # [N, T, 1, Q]
# x_y = np.take_along_axis(x, y[:, :, np.newaxis, np.newaxis], axis=2)  # [N, T, 1, M]
# pxp = np.einsum("ntkm,ntkq->ntkmq", x_y, p_uq * (1 - p_uq))
# px = np.einsum("ntkm,ntkq->ntkmq", -x_y, np.power(p_uq, 2))
# # pxp = np.einsum('ntk,ntkmq->ntkmq', ytk, pxp)
# # px = np.einsum('ntk,ntkmq->ntkmq', ytk, px)
# # pxp = np.take_along_axis(pxp, np.expand_dims(y, axis=(2,3,4)), axis=2)  # [N, T, 1, Q]
# # px = np.take_along_axis(px, np.expand_dims(y, axis=(2,3,4)), axis=2)  # [N, T, 1, Q]
# liq_pxp = np.prod(pxp, axis=(1, 2))
# liq_px = np.prod(px, axis=(1, 2))
# liw_pxp = np.einsum('nq,nmq->nmq', prob_lc, liq_pxp)
# liw_px = np.einsum('nq,nmq->nmq', prob_lc, liq_px)
#
# liw = liw_pxp + (num_lc - 1) * liw_px
# beta_grad = np.sum(liw / np.expand_dims(li, axis=(1, 2)), axis=0)  # [M, Q]
# # gradient for gamma
# wq = np.einsum('ntl,nq->nlq', h, prob_lc * (1 - prob_lc))  # [N,L,Q]
# pwq = np.einsum('nq,nlq->nlq', prob_uq, wq)  # [N,L,Q]
# A = pwq.sum(axis=0)
# A.sum()
# wqq = np.einsum("ntl,nq->lq", h, prob_lc * (1 - prob_lc))
# # pwqh = np.einsum("")
# np.sum(pwqh / np.expand_dims(li, axis=(1, 2)), axis=0)
# gamma_grad = np.sum(pwq / np.expand_dims(li, axis=(1, 2)), axis=0)
# gamma_grad


# y=y, x=x, h=h, num_lc=2


# 2.68273926, 111.53146362,   8.05300903, 803.89755249,  8.2288208

def simple_loss(gamma, h, num_lc):
    beta = gamma.reshape((-1, num_lc))
    wq = np.einsum('ntl,lq->nq', h, beta)
    prob = softmax(wq, axis=1)
    return -np.log(prob).sum()


Q = 3
lc_beta = np.array([[0, 0.2, 0.4]])
simple_loss(lc_beta, h, Q)
approx_fprime(lc_beta.reshape(-1), simple_loss, 1.4901161193847656e-08, h, Q)

wq = np.einsum('ntl,lq->nq', h, lc_beta)
prob = softmax(wq, axis=1)
li = np.prod(prob, axis=1)
hw = np.einsum('ntl,nq->nlq', h, prob * (1 - prob))
ww = np.einsum('ntl,nq->nlq', -h, np.power(prob, 2))
whw = hw + (Q - 1) * ww
np.sum(whw / np.expand_dims(prob, axis=1), axis=0)


# 2.68273926, 111.53146362,   8.05300903, 803.89755249, 8.2288208

# +++++++++++++++++++++++++++++++++++++
#           multinomial logit
# -------------------------------------
# beta0 = np.repeat(0.1234567, 8)
# cross_entropy_loss(beta=beta0, y=Y, x=X, h=H, num_lc=Q)


def gather_results(Q: int, seed=1):
    # Q:  num latent class
    assert Q > 0
    rng = np.random.default_rng(seed=seed)
    num_parameters = Q * X.shape[-1] + Q - 1
    beta0 = rng.random(size=(num_parameters,))

    result = minimize(fun=cross_entropy_loss,
                      x0=beta0, args=(Y, X, H, Q),
                      method='Nelder-Mead',  # "Nelder-Mead",  # 'Nelder-Mead',
                      options={'maxiter': 10000})
    beta = result.x
    hessian_matrix = get_hessian_matrix(loss_fn=cross_entropy_loss,
                                        beta=beta, y=Y, x=X, h=H, num_lc=Q, tol=1e-4)
    beta_sd = np.sqrt(np.diag(np.linalg.pinv(hessian_matrix)))
    t_value = beta / beta_sd
    log_loss = result.fun
    num_param = len(beta)
    AIC = 2 * (log_loss + num_param)
    BIC = 2 * log_loss + num_param * np.log(X.shape[0])
    return [beta, beta_sd, t_value, log_loss, AIC, BIC], result


# results = [gather_results(q) for q in [1, 2, 3, 4]]
A = gather_results(2, 0)
A[0]
