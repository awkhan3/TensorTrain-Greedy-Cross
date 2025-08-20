from greedy_cross import greedy_cross

def eval_tt(cores, idx):
    """Evaluate TT decomposition at a given multi-index."""
    val = cores[0][0, idx[0], :]
    for j in range(1, len(cores)):
        val = val @ cores[j][:, idx[j], :]
    return val.item()
  
def test_tt_error(cores, fun, u, num_samples=1000):
    errs = []
    for _ in range(num_samples):
        # random multi-index
        idx = [np.random.randint(0, n) for n in u]
        # evaluate true fun
        M = np.array([idx])  # shape (1, d)
        f_val = fun(M)
        # evaluate tt approx
        tt_val = eval_tt(cores, idx)
        errs.append(abs(f_val - tt_val) / (abs(f_val) + 1e-15))
    return np.max(errs), np.mean(errs)



fun = lambda M: 1.0 / (np.sum(M, axis=1) + 1)
u = [10, 12, 15, 30, 14, 15, 16]
tol = 1E-7
nswp = 1000
tt_cores = greedy_cross(u, fun, tol, nswp)


max_err, mean_err = test_tt_error(tt_cores, fun, u)
print("Max relative error:", max_err)
print("Mean relative error:", mean_err)
