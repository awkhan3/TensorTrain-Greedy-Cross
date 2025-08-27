import random
import numpy as np
rng = np.random.default_rng()

def greedy_cross(u, fun, tol, nswp):
    """
    Obtain the TT-approximation of the tensor A defined by the following entries
    A_{i_1, i_2, ..., i_n} = fun(i_1, i_2, ..., i_n),
    where 0 \le i_j \le u_j and u = [u_1, u_2, ..., u_n].
    Ideally, approximation satisfies 
    \|TT_approx(A) - A\|_{C} \le \|A\|_{C}*tol.
    The logging of the metrics of this function can be silenced by commenting out line 44. 

    :param u: list
    :param fun: vectorized function f (takes as input a matrix M \in \R^{num_evals x n} where each row 
    is an evaluation and the columns are indicies, and outputs a vector of evaluations for each row) 
    :param tol: float
    :param nswp: int
    :return: list of 3D tensors (TT-cores)
    
    This procedure implements Algorithm 2 from 
    D. Savostyanov, Quasioptimality of maximum-volume cross interpolation of tensors, Linear Algebra Applications 458, pp 217-244, 2014. 
    http://dx.doi.org/10.1016/j.laa.2014.06.006

    This code is inspired by the Matlab implementation https://github.com/oseledets/TT-Toolbox/blob/master/cross/greedy2_cross.m by S. Dolgov.

    Implementation by: Abraham Khan.
    """

    factors = []
    pre_factors = []
    dim = len(u)
    max_dx_lst = [0]*dim
    max_eval = 0
    swp = 0
    ind_selector = 0
    flags = [False]*(dim - 1)
    truth_flags = [True]*(dim - 1)

    # initialize our TT cross approximation with a random index.
    ind_left, ind_right, factors, mid_inv, ind_left_exl, ind_right_exl = init_cross_approximation(factors, u, fun)
    # instantiate the prefactors list which will store matrices that will allow us to enrich the cross-approximations
    # of the super cores more efficiently
    for i in range(len(factors) - 1):
        (r1, n1, r2) = np.shape(factors[i])
        (r2, n2, r3) = np.shape(factors[i + 1])
        f1 = factors[i].reshape(r1 * n1, r2) @ mid_inv[i][0]
        f2 = mid_inv[i][1] @ factors[i + 1].reshape(r2, n2 * r3)
        pre_factors.append([f1, f2])

    while True:
        if flags == truth_flags and ind_selector >= dim - 1:
            print("|sweep|: ", swp, "|max_error|: ", max(max_dx_lst))
            return form_tt(factors, mid_inv)
        if swp >= nswp:
            return form_tt(factors, mid_inv)
        if ind_selector >= dim - 1:
            ind_selector = 0
            swp = swp + 1

        I_le_ind = None
        I_gr_ind = None
        if ind_selector >= 1:
            I_le_ind = ind_left[ind_selector - 1]
        if ind_selector < dim - 2:
            I_gr_ind = ind_right[ind_selector + 1]

        left_factor = factors[ind_selector]
        right_factor = factors[ind_selector + 1]
        left_factor = np.reshape(left_factor, (left_factor.shape[0] * left_factor.shape[1], left_factor.shape[2]))
        right_factor = np.reshape(right_factor, (right_factor.shape[0], right_factor.shape[1] * right_factor.shape[2]))

        # pre-computed-factor: yl = left_factor @ mid_inv[ind_selector][0]
        yl = pre_factors[ind_selector][0]
        # pre-computed-factor: yr = mid_inv[ind_selector][1] @ right_factor
        yr = pre_factors[ind_selector][1]
        # if we have not exhausted all possible crosses, then add a new cross
        if yl.shape[0] != len(ind_left_exl[ind_selector]) and yr.shape[1] != len(ind_right_exl[ind_selector]):
            new_i, new_j, new_err, new_max_eval = get_new_cross(fun, I_le_ind, I_gr_ind, yl, yr, u[ind_selector],
                                                                u[ind_selector + 1], max_eval,
                                                                ind_left_exl[ind_selector], ind_right_exl[ind_selector])
            max_eval = max(max_eval, new_max_eval)
            max_dx_lst[ind_selector] = new_err / max_eval
        else: # else, the super-core is already full-rank
            flags[ind_selector] = True
            max_dx_lst[ind_selector] = 1E-16
            ind_selector = ind_selector + 1
            continue
        # if the approximate error of the super-core is less than or equal to tol, then stop
        if max_dx_lst[ind_selector] <= tol:
            flags[ind_selector] = True
            ind_selector = ind_selector + 1
            continue
        else: # otherwise, add a new cross and exclude the added cross
            flags[ind_selector] = False
            ind_left_exl[ind_selector].append(new_i)
            ind_right_exl[ind_selector].append(new_j)

        # if the tensor is a matrix (degenerate case)
        if I_le_ind is None and I_gr_ind is None:
            new_cross_i = new_i
            new_cross_j = new_j
        # if we are operating on the first super-core
        elif I_le_ind is None and I_gr_ind is not None:
            t_i, t_j = np.unravel_index(new_j, (u[ind_selector + 1], I_gr_ind.shape[0]))
            new_cross_i = new_i
            new_cross_j = np.concatenate(([t_i], I_gr_ind[t_j, :]))
        # if we are operating on the last super-core
        elif I_le_ind is not None and I_gr_ind is None:
            t_i, t_j = np.unravel_index(new_i, (I_le_ind.shape[0], u[ind_selector]))
            new_cross_i = np.concatenate((I_le_ind[t_i, :], [t_j]))
            new_cross_j = new_j
        # if we are operating on a super-core that is neither last nor first (middle super-core)
        else:
            t1_i, t1_j = np.unravel_index(new_i, (I_le_ind.shape[0], u[ind_selector]))
            t2_i, t2_j = np.unravel_index(new_j, (u[ind_selector + 1], I_gr_ind.shape[0]))
            new_cross_i = np.concatenate((I_le_ind[t1_i, :], [t1_j]))
            new_cross_j = np.concatenate(([t2_i], I_gr_ind[t2_j, :]))

        # fix the dimensions of the new cross
        new_cross_i = new_cross_i.reshape(1, np.size(new_cross_i))
        new_cross_j = new_cross_j.reshape(1, np.size(new_cross_j))
        # enrich the left and right index set by adding a new cross
        ind_left[ind_selector] = np.vstack([ind_left[ind_selector], new_cross_i])
        ind_right[ind_selector] = np.vstack([ind_right[ind_selector], new_cross_j])

        # update our factor matrices
        if I_le_ind is None:
            rn = np.reshape(np.arange(u[ind_selector]), [u[ind_selector], 1])
        else:
            rn = array_mesh(I_le_ind, np.reshape(np.arange(u[ind_selector]), [u[ind_selector], 1]))

        if I_gr_ind is None:
            nr = np.reshape(np.arange(u[ind_selector + 1]), [u[ind_selector + 1], 1])
        else:
            nr = array_mesh(np.reshape(np.arange(u[ind_selector + 1]), [u[ind_selector + 1], 1]), I_gr_ind, False)

        rn = true_vector(rn, new_cross_j, fun, False)
        factors[ind_selector] = np.hstack([left_factor, rn.reshape(np.size(rn), 1)])
        nr = true_vector(new_cross_i, nr, fun)
        factors[ind_selector + 1] = np.vstack([right_factor, nr.reshape(1, np.size(nr))])


        # update the mid_inv of the current super-core using block matrix inversion
        # the mid_inv matrix is built recursively and analytically expressed in terms of its LU factors.
        b = true_vector(ind_left[ind_selector], new_cross_j, fun, False)
        bt = true_vector(new_cross_i, ind_right[ind_selector], fun)
        b = b[:(np.size(b) - 1)]
        bt = bt[:(np.size(bt) - 1)]
        b = b.reshape(np.size(b), 1)
        bt = bt.reshape(1, np.size(bt))
        u1 = mid_inv[ind_selector][0]
        l1 = mid_inv[ind_selector][1]
        k_fac = autovecfun(fun,
                           np.array(np.append(new_cross_i, new_cross_j)).reshape(1, np.size(new_cross_j) + np.size(
                               new_cross_i))) - \
                (bt @ u1) @ (l1 @ b)

        u2 = (-1 / k_fac) * (u1 @ (l1 @ b))
        l2 = -1 * (bt @ u1) @ l1
        r = mid_inv[ind_selector][0].shape[0]
        u1n = np.zeros((r + 1, r + 1))
        l1n = np.zeros((r + 1, r + 1))
        u1n[:r, :r] = u1
        u1n[r, r] = 1 / k_fac
        u1n[0:r, r] = u2.reshape(np.shape(u1n[0:r, r]))
        l1n[:r, :r] = l1
        l1n[r, 0:r] = l2.reshape(np.shape(l1n[r, 0:r]))
        l1n[r, r] = 1.0
        mid_inv[ind_selector] = (u1n, l1n)

        # update the pre_factors list and index exclusion lists only if we are not in the degenerate case
        # where the tensor is a matrix
        l_col, l_row = np.shape(factors[ind_selector])
        r_col, r_row = np.shape(factors[ind_selector + 1])
        pre_factors[ind_selector] = [
            np.hstack([pre_factors[ind_selector][0], factors[ind_selector] @ u1n[:, r].reshape(r + 1, 1)]),
            np.vstack([pre_factors[ind_selector][1], l1n[r, :].reshape(1, r + 1) @ factors[ind_selector + 1]])]
        # update the pre-factors of the neighbouring super cores as well if they exist
        if ind_selector - 1 >= 0 and dim > 2:
            rn = rn.reshape(int(np.size(rn) / u[ind_selector]), u[ind_selector])
            rn = mid_inv[ind_selector - 1][1] @ rn
            rn = rn.reshape(np.size(rn), 1)
            pre_factors[ind_selector - 1][1] = pre_factors[ind_selector - 1][1].reshape(l_col, l_row - 1)
            pre_factors[ind_selector - 1][1] = np.hstack([pre_factors[ind_selector - 1][1], rn])
            pre_factors[ind_selector - 1][1] = pre_factors[ind_selector - 1][1].reshape(int(l_col / u[ind_selector]),
                                                                                        u[ind_selector] * l_row)
        # update the pre-factors of the neighbouring super cores as well if they exist
        if ind_selector + 1 < len(pre_factors) and dim > 2:
            nr = nr.reshape(u[ind_selector+1], int(np.size(nr) / u[ind_selector+1]))
            nr = nr @ mid_inv[ind_selector + 1][0]
            nr = nr.reshape(1, np.size(nr))
            pre_factors[ind_selector + 1][0] = pre_factors[ind_selector + 1][0].reshape(r_col - 1, r_row)
            pre_factors[ind_selector + 1][0] = np.vstack([pre_factors[ind_selector + 1][0], nr])
            pre_factors[ind_selector + 1][0] = pre_factors[ind_selector + 1][0].reshape(r_col * u[ind_selector+1],
                                                                                        int(r_row / u[ind_selector+1]))

        # reshape the factors back to TT-form as 3D tensors
        factors[ind_selector] = np.reshape(factors[ind_selector], (int(l_col / u[ind_selector]),
                                                                   u[ind_selector], l_row))
        factors[ind_selector + 1] = np.reshape(factors[ind_selector + 1],
                                               (l_row, u[ind_selector + 1], int(r_row / u[ind_selector + 1])))

        # now that the factors are changed, the positions of left_exl and right_exl must be updated as well
        # no need to update if we are in the degenerate case
        if ind_selector > 0 and dim > 2:
            temp_right_ind = np.atleast_1d(ind_right_exl[ind_selector - 1])
            temp_right_ind = temp_right_ind.reshape(-1, 1)
            temp_right_ind = np.unravel_index(temp_right_ind, [u[ind_selector], l_row - 1])
            temp_right_ind = np.ravel_multi_index(temp_right_ind, [u[ind_selector], l_row])
            ind_right_exl[ind_selector - 1] = list(np.ravel(temp_right_ind))

        if ind_selector < dim - 2 and dim > 2:
            temp_left_ind = np.atleast_1d(ind_left_exl[ind_selector + 1])
            temp_left_ind = temp_left_ind.reshape(-1, 1)
            temp_left_ind = np.unravel_index(temp_left_ind, [l_row - 1, u[ind_selector + 1]])
            temp_left_ind = np.ravel_multi_index(temp_left_ind, [l_row, u[ind_selector + 1]])
            ind_left_exl[ind_selector + 1] = list(np.ravel(temp_left_ind))

    # increment the ind_selector
        ind_selector = ind_selector + 1

def get_new_cross(fun, L, R, yl, yr, n1, n2, max_eval, ind_left_exl, ind_right_exl):
    ind1 = np.arange(n1).reshape(n1, 1)
    ind2 = np.arange(n2).reshape(n2, 1)
    ind_left_exl = np.array(ind_left_exl, dtype=int)
    ind_right_exl = np.array(ind_right_exl, dtype=int)
    # if the tensor is a matrix (degenerate case)
    if L is None and R is None:
        left = ind1
        right = ind2
    elif L is None and R is not None: # if the super-core is the first super-core
        left = ind1
        right = array_mesh(ind2, R, False)
    elif R is None and L is not None: # if the super-core is the last super-core
        left = array_mesh(L, ind1)
        right = ind2
    else: # if the super-core is a middle super-core
        left = array_mesh(L, ind1)
        right = array_mesh(ind2, R, False)

    err_diff, maxy, max_i, max_j = random_error_check(left, right, yl, yr, fun, max_eval, ind_left_exl,
                                                      ind_right_exl)
    return max_i, max_j, err_diff, maxy

def random_error_check(left, right, yl, yr, fun, max_eval, left_exl, right_exl):
    rl, nl = np.shape(yl)
    nr, rr = np.shape(yr)

    # exclude the crosses we already have
    left_mask = np.ones(rl, dtype=bool)
    if len(left_exl) > 0:
        left_mask[np.asarray(left_exl, dtype=int)] = False
    right_mask = np.ones(rr, dtype=bool)
    if len(right_exl) > 0:
        right_mask[np.asarray(right_exl, dtype=int)] = False
    left_sample = np.flatnonzero(left_mask)
    right_sample = np.flatnonzero(right_mask)

    # Use Floyd's algorithm to take samples.
    num_samples = min(np.size(left_sample), np.size(right_sample))
    samples = rng.choice(np.size(left_sample) * np.size(right_sample), num_samples, replace=False)
    ind_mat = np.array(np.unravel_index(samples, [np.size(left_sample), np.size(right_sample)]))
    left_sample = left_sample[ind_mat[0, :]]
    right_sample = right_sample[ind_mat[1, :]]
    # left samples and right samples are arrays of sample of len nr
    ind_mat1 = np.concatenate((left[left_sample, :], right[right_sample, :]), axis=1)
    y_vec1 = autovecfun(fun, ind_mat1)
    z = np.einsum('ij, ij->i', yl[left_sample, :], np.transpose(yr[:, right_sample]))
    err2 = y_vec1 - z
    ind = np.argmax(np.abs(err2))
    max_i = left_sample[ind]
    max_j = right_sample[ind]

    # randomly fix the column or row, and then maximize w.r.t it.
    if bool(random.getrandbits(1)):
        fixed_j_index = np.repeat(right[max_j, :].reshape(1, np.size(right[max_j, :])),
                                  left.shape[0], axis=0)
        ind_mat_fixed_j = np.concatenate((left, fixed_j_index), axis=1)
        y_vec2 = autovecfun(fun, ind_mat_fixed_j)
        z1 = yl @ yr[:, max_j]
        err21 = y_vec2 - z1
        # we don't want crosses we already got.
        err21[left_exl] = 0
        max_i = np.argmax(np.abs(err21))
        err_diff = abs(err21[max_i])
    else:
        fixed_i_index = np.repeat(left[max_i, :].reshape(1, np.size(left[max_i, :])),
                                  right.shape[0], axis=0)
        ind_mat_fixed_i = np.concatenate((fixed_i_index, right), axis=1)
        y_vec2 = autovecfun(fun, ind_mat_fixed_i)
        z1 = yl[max_i, :] @ yr
        err21 = y_vec2 - z1
        # we don't want crosses we already got.
        err21[right_exl] = 0
        max_j = np.argmax(np.abs(err21))
        err_diff = abs(err21[max_j])
    # pick the new max_eval
    new_max_eval = max(abs(max_eval), np.max(np.abs(y_vec1)), np.max(np.abs(y_vec2)))
    return err_diff, new_max_eval, max_i, max_j

def init_cross_approximation(factors, u, fun):
    dim = len(u)
    ind_left = []
    ind_right = []
    left_exl = []
    right_exl = []
    num_crosses = 1
    non_vec_fun = lambda X: fun(np.array(X).reshape(1, np.size(X)))
    idx = np.array([rng.integers(0, uk - 1) for uk in u], dtype=np.int64)

    # start with random index
    for i in range(dim):
        ind_left.append(idx[:i+1].reshape(1, i+1 ))
        ind_right.append(idx[i+1:].reshape(1, dim - (i + 1)))
    ind_left.pop(-1)
    ind_right.pop(-1)

    # now maximize
    sweeps = 2
    for _ in range(sweeps):
        # backward pass
        for ax in range(dim - 1, -1, -1):
            n_ax = u[ax]
            J = np.broadcast_to(idx, (n_ax, dim)).copy()
            J[:, ax] = np.arange(n_ax)
            vals = autovecfun(fun, J)
            j_opt = int(np.argmax(np.abs(vals)))
            if j_opt != idx[ax]:
                idx[ax] = j_opt
        # forward pass
        for ax in range(dim):
            n_ax = u[ax]
            J = np.broadcast_to(idx, (n_ax, dim)).copy()
            J[:, ax] = np.arange(n_ax)
            vals = autovecfun(fun, J)
            j_opt = int(np.argmax(np.abs(vals)))
            if j_opt != idx[ax]:
                idx[ax] = j_opt

    mid_inv = []
    for ind_selector in range(dim - 1):
        left_ind = ind_left[ind_selector]
        right_ind = ind_right[ind_selector]
        A_cross = np.zeros((left_ind.shape[0], right_ind.shape[0]))
        A_cross[0, 0] = 1/non_vec_fun((list(left_ind[0][:]) + list(right_ind[0][:])))
        mid_inv.append((np.array([1]).reshape(1,1), A_cross))

    for i in range(dim - 1):
        left_exl.append([ind_left[i][0, -1]])
        right_exl.append([ind_right[i][0, 0]])

    nr = array_mesh(np.arange(u[0]).reshape(u[0], 1), ind_right[0], False)
    C1 = autovecfun(fun, nr)
    C1 = C1.reshape(1, u[0], num_crosses)
    rn = array_mesh(ind_left[dim - 2], np.arange(u[dim - 1]).reshape(u[dim - 1], 1))
    Cn = autovecfun(fun, rn)
    Cn = Cn.reshape(num_crosses, u[dim - 1], 1)
    factors.append(C1)
    for ind_selector in range(1, dim - 1):
        left_ind = ind_left[ind_selector - 1]
        right_ind = ind_right[ind_selector]
        C = np.zeros((num_crosses, u[ind_selector], num_crosses))
        for i in range(u[ind_selector]):
            C[0, i, 0] = non_vec_fun((list(left_ind[0, :]) + [i] + list(right_ind[0, :])))
        factors.append(C)
    factors.append(Cn)

    return ind_left, ind_right, factors, mid_inv, left_exl, right_exl


def array_mesh(left, right, switch=True):
    if switch:
        r = left.shape[0]
        n = right.shape[0]
        left = np.repeat(left, n, axis=0)
        right = np.tile(right, (r, 1))
        return np.hstack([left, right])
    else:
        n = left.shape[0]
        r = right.shape[0]
        left = np.repeat(left, r, axis=0)
        right = np.tile(right, (n, 1))
        return np.hstack([left, right])


def true_vector(L, R, fun, switch=True):
    if switch:
        ind = L
        ind = np.repeat(ind, R.shape[0], axis=0)
        return autovecfun(fun, np.hstack([ind, R]))
    else:
        ind = R
        ind = np.repeat(ind, L.shape[0], axis=0)
        return autovecfun(fun, np.hstack([L, ind]))


def autovecfun(fun, J, vec_flag=True):
    if vec_flag:
        return fun(J)

    s1 = np.shape(J)[0]
    y = np.zeros(s1)
    for i in range(s1):
        y[i] = fun(*tuple(J[i, :]))
    return y

def form_tt(factors, mid_inv):
    p = (np.array([1]).reshape((1, 1)), np.array([1]).reshape((1, 1)))
    mid_inv.insert(0, p)
    mid_inv.append(p)
    cores = []
    for i in range(0, len(factors)):
        core = factors[i]
        r1, n, r2 = np.shape(factors[i])
        core = mid_inv[i][1] @ np.reshape(core, (r1, n * r2))
        core = np.reshape(core, (n * r1, r2)) @ mid_inv[i + 1][0]
        core = np.reshape(core, (r1, n, r2))
        cores.append(core)
    return cores
