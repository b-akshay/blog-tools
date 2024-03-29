"""
Implements label propagation and related methods, as described in: 
- https://akshay.bio/blog/graph-prediction-diffusion/
"""


import numpy as np, scipy


def harmonic_extension(
    labeled_signal, # n-matrix with labeled rows set to their fixed values, and unlabeled rows set to arbitrary values.
    adj_mat, # (n x n) adjacency matrix
    labeled_ndces, # Boolean mask indicating which cells are in the labeled set.
    method='iterative', 
    eps_tol=0.01 # Min. relative error in consecutive iterations of F before stopping (normally <= 20 iterations)
):
    """
    Given a graph and a continuous label signal over labeled_ndces, 
    impute the rest of the signal with its harmonic extension (hard clamping). 
    Returns n-vector predicting signal over the entire graph.
    """
    labels = labeled_signal[labeled_ndces]
    if scipy.sparse.issparse(labels):
        labels = labels.toarray()
    num_labeled = np.sum(labeled_ndces)
    num_unlabeled = adj_mat.shape[0] - num_labeled
    pmat = scipy.sparse.diags(1.0/np.ravel(adj_mat.sum(axis=0))).dot(adj_mat)
    p_uu = pmat[~labeled_ndces, :][:, ~labeled_ndces]
    p_ul = pmat[~labeled_ndces, :][:, labeled_ndces]
    inv_sofar = p_ul.dot(labels)
    if method == 'iterative':
        # Power series (I - P_uu)^{-1} = I + P_uu + P_uu^2 + ...
        cummat = p_ul.dot(labels)
        cumu = []
        stop_crit = False
        while not stop_crit:
            cummat = p_uu.dot(cummat)
            rel_err = np.square(cummat).sum()/np.square(inv_sofar).sum()
            inv_sofar = inv_sofar + cummat
            cumu.append(inv_sofar)
            if rel_err <= eps_tol:
                stop_crit = True
        cumu.append(inv_sofar)
        # Add unlabeled indices back into their respective places.
        for i in range(len(cumu)):
            to_add = np.zeros(labeled_signal.shape)
            to_add[labeled_ndces] = labels
            to_add[~labeled_ndces] = cumu[i]
            cumu[i] = to_add
        return cumu
    elif method == 'direct':
        toret = scipy.sparse.linalg.lsmr(scipy.sparse.identity(num_unlabeled) - p_uu, inv_sofar)
        return toret[0]
        return toret[0]


def label_propagation(
    labeled_signal, # (n x |Y|) matrix with unlabeled rows set to arbitrary values.
    adj_mat, # (n x n) adjacency matrix
    labeled_ndces, # Boolean mask indicating which cells are in the labeled set.
    param_alpha=0.8, 
    method='iterative', 
    eps_tol=0.01   # Min. relative error in consecutive iterations of F before stopping (normally <= 20 iterations)
):
    """
    From Zhou et al. 2003 "Learning with local and global consistency".
    Returns an n-vector of real-valued relative confidences.
    """
    labeled_signal[~labeled_ndces, :] = 0
    dw_invsqrt = scipy.sparse.diags(
        np.reciprocal(np.sqrt(np.ravel(adj_mat.sum(axis=0))))
    )
    R = dw_invsqrt.dot(adj_mat).dot(dw_invsqrt)
    F = labeled_signal.copy()
    if scipy.sparse.issparse(F):
        F = F.toarray()
    cumu = []
    if method == 'iterative':
        stop_crit = False
        while not stop_crit:
            F_new = np.array((param_alpha*R.dot(F)) + ((1-param_alpha)*np.array(labeled_signal)))
            rel_err = np.square(F_new - F).sum()/np.square(F_new).sum()
            F = F_new
            cumu.append(F)  # np.argmax(F, axis=1)
            print(rel_err)
            if rel_err <= eps_tol:
                stop_crit = True
        cumu.append(F)  # np.argmax(F, axis=1)
        return cumu
    elif method == 'direct':
        return scipy.sparse.linalg.lsmr(scipy.sparse.identity(R.shape[0]) - param_alpha*R, labeled_signal)


