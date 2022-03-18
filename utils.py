import numpy as np


def softmax(u, axis):
    # stable version
    M = np.max(u, axis=axis, keepdims=True)
    up = np.exp(u - M)
    return up / up.sum(axis=axis, keepdims=True)


def approximate_hessian(loss_fn,
                        beta: np.array,
                        hi: float, ei: np.array, hj: float, ej: np.array,
                        y, x, *args, **kwargs):
    # partial f^2 / partial x_i partial x_j = f(x+hi*ei + hj*ej) - f(x+hi*ei - hj*ej) - f(x-hi*ei + hj*ej) \
    # + f(x-hi*ei - hj*ej) / [4hi*hj]
    # central difference
    assert len(ei) == len(beta) & len(ej) == len(beta)
    f_pi_pj = loss_fn(beta + hi * ei + hj * ej, y=y, x=x, *args, **kwargs)
    f_pi_mj = loss_fn(beta + hi * ei - hj * ej, y=y, x=x, *args, **kwargs)
    f_mi_pj = loss_fn(beta - hi * ei + hj * ej, y=y, x=x, *args, **kwargs)
    f_mi_mj = loss_fn(beta - hi * ei - hj * ej, y=y, x=x, *args, **kwargs)
    return (f_pi_pj - f_pi_mj - f_mi_pj + f_mi_mj) / (4 * hi * hj)


def get_hessian_matrix(loss_fn,
                       beta: np.array,
                       y, x,
                       tol=1e-4,
                       *args, **kwargs):
    base_vectors = np.eye(len(beta))
    hessian_matrix = np.zeros((len(beta), len(beta)))
    for i in range(len(beta)):
        e_i = base_vectors[i]
        hi = abs(beta[i] * tol)
        for j in range(len(beta)):
            e_j = base_vectors[j]
            hj = abs(beta[j] * tol)
            hessian_matrix[i, j] = approximate_hessian(loss_fn=loss_fn,
                                                       beta=beta, hi=hi, ei=e_i, hj=hj, ej=e_j,
                                                       y=y, x=x, *args, **kwargs)
    # sd = np.sqrt(np.diag(np.linalg.pinv(hessian_matrix)))
    return hessian_matrix


def load_data():
    dat = np.loadtxt("data.txt")
    # X: [N, 4], ID, Time, var1_choice_1, var1_choice_2
    # ID: 1-300, Time: 1-10
    dat_x = np.delete(dat, 2, axis=1)
    # Y: [N,1], Choice ID, 1 or 2
    dat_y = dat[:, 2]
    Y = (dat_y - 1).astype(int)
    # convert data to [N*T, K, M], M: constant, price
    X = np.zeros((dat_x.shape[0], Y.max() + 1, 2))
    X[:, 0, 0] = 1  # first choice is a baseline
    X[:, 0, 1] = dat_x[:, 2]
    X[:, 1, 1] = dat_x[:, 3]
    return X, Y
