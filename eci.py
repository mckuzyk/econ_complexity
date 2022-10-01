"""
Functions for computing the economic comlexity index and related quantiities.
"""


import numpy as np
from tqdm import tqdm


def specialization_matrix(Xcp):
    """
    Calculate the specialization matrix from Xcp, a matrix that connects 
    locations to activities.
    """

    X = np.sum(Xcp)
    Xc = np.sum(Xcp, axis=1)
    Xp = np.sum(Xcp, axis=0)

    R = np.zeros(np.shape(Xcp))
    rows, cols = np.shape(R)
    print('Populating specialization matrix.')
    for i in tqdm(range(rows), total=rows):
        for j in range(cols):
            if Xc[i] == 0 or Xp[j] == 0:
                R[i][j] = 0
                continue
            R[i][j] = X * Xcp[i][j] / Xc[i] / Xp[j]
    
    return R


def binary_specialization_matrix(R):
    """
    Calculate the binary specialization matrix M from the specialization 
    matrix R.
    """

    M = np.zeros(np.shape(R), dtype=int)
    M[R >= 1] = 1

    return M


def diversity(M):
    return np.sum(M, axis=1)


def ubiquity(M):
    return np.sum(M, axis=0)


def filter_M(M):
    """
    Remove rows and columns that contain only zeros.
    """
    rows, cols = np.shape(M)
    col_mask = np.where(ubiquity(M) == 0, True, False)
    M = np.delete(M, col_mask, 1)
    row_mask = np.where(diversity(M) == 0, True, False)
    M = np.delete(M, row_mask, 0)

    return M, np.invert(row_mask), np.invert(col_mask)


def M_tilde(M):
    """
    M^\tilde can be computed as D^{-1}MU^{-1}M^T, where D and U are diagonal
    matrices constructed from the diversity and ubiquity.
    """
    D_inv = np.diag(1/diversity(M))
    U_inv = np.diag(1/ubiquity(M))
    return D_inv @ M @ U_inv @ M.transpose()


def M_hat(M):
    D_inv = np.diag(1/diversity(M))
    U_inv = np.diag(1/ubiquity(M))
    return U_inv @ M.transpose() @ D_inv @ M


def activity_proximity(M):
    u = ubiquity(M)
    phi = M.transpose() @ M
    rows, cols = np.shape(phi)
    for i in range(rows):
        for j in range(cols):
            phi[i][j] = phi[i][j] / max(u[i],u[j])
    return phi


def relatedness_density(M):
    phi = activity_promximity(M)
    phi_p = np.sum(phi, axis=1)
    omega = M @ phi.transpose()
    rows, cols = np.shape(omega)
    for i in range(rows):
        for j in range(cols):
            omega[i][j] = omega[i][j] / phi_p[j]
    return omega


def eci(M_tilde):
    """Economic complexity index computed from Mtilde matrix.
    
    Computes eigenvector corresponding to second largest eigenvlaue of M_tilde.
    """
    w, v = np.linalg.eig(M_tilde)
    order = w.argsort()[::-1]
    for i in range(len(order)):
        if w[i] != w[0]:
            break
    return np.real(v[:, order[i]])


def pci(M_hat):
    """Product complexity index computed from Mhat matrix.

    Computes eigenvector corresponding to second largest eigenvlaue of M_hat.
    """
    w, v = np.linalg.eig(M_hat)
    order = w.argsort()[::-1]
    for i in range(len(order)):
        if w[i] != w[0]:
            break
    return np.real(v[:, order[i]])


def normalized_eci(v):
    """Normalized version of eci."""
    return (v - np.average(v)) / np.std(v)


def normalized_pci(v):
    """Normalized version of pci."""
    return (v - np.average(v)) / np.std(v)


def hhi(Xcp):
    """Compute the Herfindahl-Hirschman index."""
    Xcp = Xcp.astype(np.float64)
    total_revenues = np.sum(Xcp, axis=1)
    total_revenues[total_revenues == 0] = np.inf
    return np.sum(Xcp**2, axis=1) / total_revenues**2


def normalized_hhi(Xcp):
    """Compute the normalized Herfindahl-Hirschman index.

    HHI falls in the range [1/N, 1], where N is the number of entities in the
    market.  The normalized variant of HHI falls in the range [0,1], independent
    of the number of entities.  The nomralization is defined such that the
    nhhi = 0 if N = 0 and nhhi = 1 if N = 1.
    """
    _hhi = hhi(Xcp)
    market_participants = np.sum(np.where(Xcp==0, 0, 1), axis=1)
    n = np.zeros(len(_hhi)) 
    for i, val in _hhi:
        if market_participants == 0:
            n[i] = 0
            continue
        if market_participant == 1:
            n[i] = 1
            continue
        n[i] = val
    return n


def get_values(Xcp):
    """Compute eci, pci, diversity, ubiquity from Xcp.


    Input
    ------
    Xcp: numpy 2D array
        Xcp gives the volume of p taking place in c


    Returns
    -------
    eci:
        The "economic complexity index", ie complexity of the rows of Xcp
    pci:
        The "product complexity index", ie complexity of the cols of Xcp
    diversity:
        Number of column entries that participate in a row
    ubiquity:
        Number of row entries that participate in a col
    row_mask: array(bool)
        row_mask[i] = True if row i was kept (after dropping rows of zeros)
    col_mask: array(bool)
        col_mask[i] = True if col i was kept (after dropping cols of zeros)
    """

    # Compute binary specialization matrix
    M = binary_specialization_matrix(specialization_matrix(Xcp))

    # Filter out rows and cols that are all zeros
    M, row_mask, col_mask = filter_M(M)

    # Diversity and ubiquity
    u = ubiquity(M)
    d = diversity(M)

    # Eigenvalues of second largest eigenvector of Mtilde are eci
    Mtilde = M_tilde(M)
    # Eigenvalues of second largest eigenvector of Mhat are pci
    Mhat = M_hat(M)

    # eci = complexity of rows
    kc = eci(Mtilde)
    kc = normalized_eci(kc)

    # pci = complexity of cols
    pc = pci(Mhat)
    pc = normalized_pci(pc)

    return kc, pc, d, u, row_mask, col_mask
