import numpy as np, scipy
import scipy.stats

def Chat19_coef_1(xs, ys):
    """
    Univariate Chatterjee coefficient.
    """
    ranks = scipy.stats.rankdata(ys[np.argsort(xs)])
    sumdiff = np.sum(np.abs(np.diff(ranks)))
    return 1 - sumdiff*3.0/(len(ranks)**2 - 1)


# Version of the univariate Chatterjee coefficient with ties allowed.
def Chat19_coef(xs, ys):
    ranks = scipy.stats.rankdata(ys[np.argsort(xs)])
    sumdiff = np.sum(np.abs(np.diff(ranks)))
    lks = len(ranks) + 1 - ranks
    denom = np.mean(np.multiply(len(ranks)-lks, lks))
    ret = 1 - (sumdiff/(2*denom))
    return ret


from sklearn.neighbors import NearestNeighbors

def AC21_coef(xs, ys, cond_covariates=None):
    """
    Multivariate version of the dependence coefficient. From Azadkia & Chatterjee, 2021.
    xs: (n x d1) array of "independent variable" covariates, where n is the number of samples.
    ys: n-dimensional vector of "dependent variable" covariate values.
    cond_covariates: (n x d2) array of "conditional" covariates, where n is the number of samples.
    """
    ranks = scipy.stats.rankdata(ys, method='max')
    n = len(ys)
    if cond_covariates is not None:
        #print(cond_covariates.shape)
        if len(xs.shape) == 1:
            xs = np.reshape(xs, (-1, 1))
        #print(cond_covariates.shape)
        joint_data = np.hstack((xs, cond_covariates))
        distances_cond, indices_cond = NearestNeighbors(n_neighbors=2).fit(cond_covariates).kneighbors(cond_covariates)
        distances_joint, indices_joint = NearestNeighbors(n_neighbors=2).fit(joint_data).kneighbors(joint_data)
        cond_ranks = np.minimum(ranks[indices_cond[:, 1]], ranks)
        joint_ranks = np.minimum(ranks[indices_joint[:, 1]], ranks)
        denominator = (1.0/n)*np.mean(ranks - cond_ranks)
        if denominator == 0:
            return None
        numerator = (1.0/n)*np.mean(joint_ranks - cond_ranks)
        return numerator/denominator
    else:
        xs = xs.reshape(-1, 1)
        complement_ranks = scipy.stats.rankdata( -ys, method='max')
        distances_data, indices_data = NearestNeighbors(n_neighbors=2).fit(xs).kneighbors(xs)
        data_ranks = np.minimum(ranks[indices_data[:, 1]], ranks)
        numerators = n*data_ranks - np.square(complement_ranks)
        numer = (1.0/(n*n))*np.mean(numerators)
        denominators = np.multiply(complement_ranks, n - complement_ranks)
        denom = (1.0/(n*n))*np.mean(denominators)
        return numer/denom


def LH21_coef(xs, ys, num_nbrs=1):
    """
    Modified univariate version of the dependence coefficient. From Lin & Han, 2021.
    xs: n-dimensional vector of "independent variable" covariates, where n is the number of samples.
    ys: n-dimensional vector of "dependent variable" covariate values.
    num_nbrs: number of (right) nearest neighbors to use. An integer >= 1.
    """
    ranks = scipy.stats.rankdata(ys, method='max')
    n = len(ys)
    jmi_arrs = []
    for k in range(1, num_nbrs+1):
        x_ndces = list(np.argsort(xs))
        a_arr = x_ndces.copy()
        x_ndces[:-k] = x_ndces[k:]
        b_arr = x_ndces
        jmi_arrs.append( np.array(b_arr)[np.argsort(a_arr)] )
    jmi_arrs = np.stack(jmi_arrs)
    ctrl_indices = np.tile(np.arange(len(ranks)), (jmi_arrs.shape[0], 1))
    summed_rank_vars = np.minimum(ranks[ctrl_indices], ranks[jmi_arrs]).sum()
    denom = (n+1)*((n*num_nbrs) + (num_nbrs*(num_nbrs+1)/4))
    return (6*summed_rank_vars/denom) - 2


def feat_selection(signal, featmat, num_features=10):
    """
    Forward greedy feature selection, with the conditional Chatterjee coefficient.
    """
    itime = time.time()
    selected_feats = []
    selected_featmat = None
    unselected_feats = list(range(featmat.shape[1]))
    while len(selected_feats) < num_features:
        max_var_explained = 0
        feat_to_add = 0
        for feat_ndx in unselected_feats:
            feat_arr = featmat[:, feat_ndx]
            feat_coeff = AC21_coef(feat_arr, signal, cond_covariates=selected_featmat)
            # Add to $S$ whichever feature $X_m \in U$ maximizes $T_n (X_m , Y \mid S)$.
            if feat_coeff > max_var_explained:
                feat_to_add = feat_ndx
                max_var_explained = feat_coeff
                print(feat_ndx, feat_coeff)
        unselected_feats.remove(feat_to_add)
        selected_feats.append(feat_to_add)
        selected_featmat = featmat[:, selected_feats]
        print(selected_feats, time.time() - itime)
    return selected_feats
