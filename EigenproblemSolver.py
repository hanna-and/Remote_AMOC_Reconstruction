import numpy as np
from scipy.linalg import eig

def stable_modes(x, N2):
    """
    Solve:
        d²h/dz² + (N² / c²) h = 0
    using a non-uniform grid (interior points only).

    Parameters
    ----------
    x : array (n+2,)
        full vertical grid INCLUDING boundaries
    N2 : array (n,)
        buoyancy frequency at interior points

    Returns
    -------
    c_modes : array
        phase speeds (m/s)
    eigenvectors : array (n, n_modes)
        vertical modes (interior points only)
    eigenvalues : array
        eigenvalues (lambda = 1/c²)
    """

    x = np.asarray(x)
    N2 = np.asarray(N2)

    Nx = len(N2)  # number of interior points

    # grid spacing
    h_space = np.diff(x)        # length n+1
    h_plus  = h_space[1:]       # length n
    h_minus = h_space[:-1]      # length n

    # coefficients 
    alpha =  2 / (h_minus * (h_minus + h_plus))
    beta  = -2 / (h_minus * h_plus)
    gamma =  2 / (h_plus * (h_minus + h_plus))

    # second derivative matrix A
    A = (
        np.diag(beta) +
        np.diag(alpha[1:], k=-1) +
        np.diag(gamma[:-1], k=1)
    )

    # buoyancy matrix B - implciityly assuming that N2 and h lie on the same grid 
    B = np.diag(N2)

    # solve eigenvalue problem
    # A h = - lambda B h; so consider (-A) h = lambda B h
    eigenvalues, eigenvectors = eig(-A, B)

    # keep real eigenvalues
    real_mask = np.abs(eigenvalues.imag) < 1e-10
    eigenvalues = eigenvalues[real_mask].real
    eigenvectors = eigenvectors[:, real_mask].real

    # keep physical modes where eigenvalue is greater than zero 
    physical_mask = eigenvalues > 0
    eigenvalues = eigenvalues[physical_mask]
    eigenvectors = eigenvectors[:, physical_mask]

    # phase speed 
    c_modes = 1 / np.sqrt(eigenvalues)

    # sort modes fast to slow 
    idx = np.argsort(eigenvalues)
    c_modes = c_modes[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]

    # applying boundary conditions 
    n_modes = eigenvectors.shape[1]
    eigenvectors_full = np.zeros((len(x), n_modes))
    
    # Set boundary conditions: h=0 at surface (index 0) and bottom (index -1)
    eigenvectors_full[0, :] = 0.0      # surface boundary
    eigenvectors_full[1:-1, :] = eigenvectors  # interior points
    eigenvectors_full[-1, :] = 0.0     # bottom boundary

    return c_modes, eigenvectors_full, eigenvalues
