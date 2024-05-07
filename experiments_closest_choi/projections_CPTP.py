import numpy as np 
import cvxpy as cp 
import timeit
import time
import scipy as sp
import collections


def choi_matrix(U):
    """
    Calculate the Choi matrix representation of a given quantum gate.

    Args:
    - U (array-like): Unitary matrix representing the quantum gate.

    Returns:
    - dens_out (array-like): Choi matrix of the quantum gate.

    This function takes a unitary matrix representing a quantum gate and computes its Choi matrix,
    which characterizes the gate's action on bipartite quantum systems.
    Note: The input unitary matrix must be square and have equal row and column dimensions.
    """

    dim = U.shape[0]  
    diag_r = np.eye(dim, dim)  
    diag_a = np.eye(dim, dim)

    psi = np.zeros((dim**2, 1))  
    for k in range(dim):
        psi = psi + np.kron(diag_r[:, k], diag_a[:, k]).reshape(-1, 1)
    
    psi = psi / np.linalg.norm(psi)  
    dens = np.kron(psi.conj().T, psi)  
    # Construct unitary matrix acting on the first part of the state
    U_comp = np.kron(U, diag_r)  
     # Apply the unitary transformation to the density matrix
    dens_out = U_comp @ dens @ U_comp.conj().T 
    
    return dens_out


def initialization(qubits, p): 
    """
    Creates a noisy Choi matrix with noise level p.

    Args:
    - qubits (int): Number of qubits for the quantum operation.
    - p (float):  Probability of noise in the generated quantum state.

    Returns:
    - dens_out (array-like): Choi matrix of the quantum gate.

    This function initializes a random unitary matrix and generates a noisy quantum state based on it.
    It first generates a random unitary matrix using QR decomposition of a random matrix with normal
    distribution. Then, it computes the Choi matrix of the unitary matrix to represent the quantum 
    operation and adds Gaussian noise to the Choi state. The amount of noise
    added is controlled by the parameter 'p'.
    """

    dim = int(2 ** qubits)
    d = int(2 ** (qubits / 2))
    # random unitary 
    random_matrix = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) / np.sqrt(2)
    Q, R = np.linalg.qr(random_matrix)
    U = np.dot(Q, np.diag(np.exp(1j * np.angle(np.diag(R)))))
    # Compute the Choi matrix of the random unitary
    choiState = choi_matrix(U)
    
    noise = np.random.normal(0, 1, dim * dim).reshape((dim, dim))
    noise1 = np.random.normal(0, 1, dim * dim).reshape((dim, dim))
    noise2 = (noise + 1j * noise1)
    noise3 = noise2 + np.conjugate(np.transpose(noise2))
    noise3 = noise3 / np.trace(noise3)
    
    rhoInput =  p * noise3 + (1-p) * choiState

    return rhoInput

    
def proj_pos(matrix):
    """
    Project a Hermitian matrix onto the positive semidefinite cone.

    Args:
    - matrix (array-like): Matrix to be projected.

    Returns:
    - pos_matrix (array-like): Positive semidefinite matrix obtained after projection.

    This function projects a Hermitian matrix onto the positive semidefinite cone.
    It first symmetrizes the input matrix by averaging it with its conjugate transpose.
    Then, it computes the singular value decomposition (SVD) of the symmetrized matrix.
    Using the SVD, it constructs a positive semidefinite matrix by eliminating negative eigenvalues.
    Finally, it symmetrizes the obtained matrix to ensure it remains Hermitian.
    """
    
    B = (matrix + matrix.conj().T) / 2 
    _, s, V = np.linalg.svd(B) 
    # Construct positive semidefinite matrix
    H = np.dot(V.conj().T, np.dot(np.diag(s), V)) 
    A2 = (B + H) / 2 
    # Ensure the matrix remains Hermitian
    pos_matrix = (A2 + A2.conj().T) / 2  

    return pos_matrix


def proj_tp(rho): 
    """
    Project a density matrix onto the space of physically admissible states under partial trace.

    Args:
    - rho (array-like): Matrix to be projected.

    Returns:
    - rhoTP (array-like): Density matrix projected onto the space of physically admissible states.

    This function projects a density matrix onto the space of trace-preserving matrices.
    It reshapes the input density matrix to a four-dimensional tensor representing a bipartite system.
    Then, it computes the partial trace over one subsystem to obtain the reduced density matrix.
    The function constructs the physically admissible state by subtracting the partial trace from a maximally
    mixed state of the same subsystems. Finally, it returns the projected density matrix.
    """

    dim, _ = rho.shape
    d = int(np.sqrt(dim))
    qubits = int(np.log2(dim))
    rhoRes = rho.reshape([d, d, d, d])
    rhoA = np.einsum('ijik->jk', rhoRes) # partial trace 
    rhoTP = rho + np.kron((1/d) * np.eye(int(2 ** (qubits/2))), (1/d) * np.eye(int(2 ** (qubits/2)))) - np.kron((1/d) * np.eye(int(2 ** (qubits/2))), rhoA) 
    #print("Check if rhoTP is TP: {}".format(np.isclose(np.einsum('ijik->jk',  rhoTP.reshape([8, 8, 8, 8])), np.eye(int(2 ** (qubits/2))) / d), atol = 1e-8))
    return rhoTP


def exactSDP(rhoInput):
    """
    Solve the exact QDT problem using Semidefinite Programming (SDP).

    Args:
    - rhoInput (array-like): Target density matrix to be reconstructed.

    Returns:
    - rho.value (array-like): Reconstructed density matrix.
    - outTime - initTime (float): Time taken for the reconstruction process.

    This function solves the exact QPT problem using Semidefinite Programming (SDP).
    It formulates the problem as an SDP by defining the density matrix 'rho' as a complex-valued
    variable and setting up appropriate constraints. The constraints enforce hermiticity, positivity,
    and trace preservation of the reconstructed density matrix. The objective function minimizes the
    Frobenius norm of the difference between the reconstructed and target density matrices.
    The SDP problem is then solved using the SCS solver with a specified tolerance.
    The function returns the reconstructed density matrix and the time taken for the reconstruction.
    """

    initTime = timeit.default_timer()
    dim, _ = rhoInput.shape
    qubits = int(np.log2(dim))
    ptsubsys =  0
    subs = [int(2 ** (qubits/2)), int(2 ** (qubits/2))]
    rho = cp.Variable((dim, dim), complex = True)
    constraints = [rho.H == rho] # hermitian
    constraints += [rho >> 0]
    constraints += [cp.partial_trace(rho, dims = subs, axis = ptsubsys) == np.eye(int(2 ** (qubits/2))) / cp.sqrt(dim)] # trace preserving
    obj = cp.norm(rho - rhoInput, "fro") ** 2 # objective function to minimise
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver = "SCS", eps=1e-8)
    outTime = timeit.default_timer()
    return rho.value, outTime - initTime 


def two_stage_solution(matrix, pos=True): 
    """
    Perform two-stage solution algorithm.

    Args:
    - matrix (array-like): Input matrix.
    - pos (bool): Flag to indicate whether to enforce positivity (default is True).

    Returns:
    - rhoOut (array-like): CPTP matrix.
    - finalTime - initTime (float): Time taken for the optimization process.

    This function implements a two-stage solution for optimizing a density matrix.
    In the first stage, if specified, it enforces positivity of the input matrix.
    Next, it projects the matrix onto the trace-preserving set via unitary transformations.
    """    

    initTime = timeit.default_timer()
    dim, _ = matrix.shape
    d = int(np.sqrt(dim))
    qubits = int(np.log2(dim))
    if pos:
        pass
    else:
        matrix = proj_pos(matrix)
        
    rhoResh = matrix.reshape([d, d, d, d])
    rhoA = np.einsum('ijik->jk', rhoResh) # partial trace
    U = np.kron(np.eye(int(2 ** (qubits/2))), sp.linalg.inv(sp.linalg.sqrtm(d * rhoA)))
    rhoOut = U @ matrix @ U.conj().T
    finalTime = timeit.default_timer()
    
    return rhoOut, finalTime - initTime


def cholesky_based_approximation(matrix, pos=True): 
    """
    Perform Cholesky-based approximation to obtain a CPTP matrix.

    Args:
    - matrix (array-like): Input matrix.
    - pos (bool): Flag to indicate whether to enforce positivity (default is True).

    Returns:
    - rhoOut (array-like): CPTP matrix.
    - finalTime - initTime (float): Time taken for the optimization process.

    This function implements the Cholesky-based approximation.
    In the first stage, if specified, it projects the input matrix onto the set of density 
    matrices via Smolin algorithm. Next, it projects the matrix onto the trace-preserving set 
    via unitary transformations.
    """  

    initTime = timeit.default_timer()
    dim, _ = matrix.shape
    d = int(np.sqrt(dim))
    qubits = int(np.log2(dim))
    if pos:
        pass
    else:
        matrix = proj_CP_threshold(matrix, free_trace=False, full_output=False, thres_least_ev=False)
        
    rhoResh = matrix.reshape([d, d, d, d])
    rhoA = np.einsum('ijik->jk', rhoResh) # partial trace
    U = np.kron(np.eye(int(2 ** (qubits/2))), sp.linalg.inv(sp.linalg.sqrtm(d * rhoA)))
    rhoOut = U @ matrix @ U.conj().T
    finalTime = timeit.default_timer()
    
    return rhoOut, finalTime - initTime


def dykstra(matrix, order, iters, tol):
    """
    Perform Dykstra's alternating projection algorithm to approximate CPTP matrices.

    Args:
    - matrix (array-like): Input matrix.
    - order (str): Order of projection (either "TPP" or "PTP").
    - iters (int): Maximum number of iterations.
    - tol (float): Tolerance for convergence.

    Returns:
    - pos_matrix (array-like): Approximated CP and TP matrix.
    - finalTime (float): Time taken for the computation.
    - it-1 (int): Number of iterations performed.

    This function implements Dykstra's alternating projection algorithm to approximate CPTP matrices. 
    It alternates between projecting onto the set of positive semidefinite matrices
    and the trace-preserving set. The algorithm terminates when either the maximum
    number of iterations is reached or the change in the projection variables is below the specified tolerance.
    The function returns the approximated CPTP matrix, the computation time, and the number
    of iterations performed.
    """
    
    initTime = timeit.default_timer()
    dim, _ = matrix.shape
    q = np.zeros((dim, dim), dtype=complex)
    p = np.zeros((dim, dim), dtype=complex)
    it = 1
    cI = 100
    
    if order=="TPP":
        pos_matrix = matrix.copy()
        p_copy = p.copy()
        q_copy = q.copy()
        while it <= iters and cI >= tol:
            tp_matrix = proj_tp(pos_matrix - p)
            p = tp_matrix + p - pos_matrix  
            pos_matrix = proj_CP_threshold(tp_matrix - q,  free_trace=False, full_output=False, thres_least_ev=False)
#             pos_matrix = proj_pos(tp_matrix - q)
            q = pos_matrix + q - tp_matrix
            cI = (np.linalg.norm(p - p_copy, "fro")**2 
                  + np.linalg.norm(q - q_copy, "fro")**2)
            p_copy = p.copy()
            q_copy = q.copy()
            it += 1
        finalTime = timeit.default_timer()
        return pos_matrix, finalTime, it-1
    
    if order=="PTP":
        tp_matrix = matrix.copy()
        p_copy = p.copy()
        q_copy = q.copy()
        while it <= iters and cI >= tol:
            pos_matrix = proj_CP_threshold(tp_matrix - p,  free_trace=False, full_output=False, thres_least_ev=False)
#             pos_matrix = proj_pos(tp_matrix - p)
            p = pos_matrix + p - tp_matrix
            tp_matrix = proj_tp(pos_matrix - q)
            q = tp_matrix + q - pos_matrix
            cI = (np.linalg.norm(p - p_copy, "fro")**2 
                  + np.linalg.norm(q - q_copy, "fro")**2)
            p_copy = p.copy()
            q_copy = q.copy()
            it += 1
        finalTime = timeit.default_timer()
        return tp_matrix, finalTime-initTime, it-1

    
def dykstra_cholesky_based_approximation(matrix, iters, tol): 
    """
    Perform Dykstra's alternating projection algorithm followed by Cholesky-based approximation.

    Args:
    - matrix (array-like): Input matrix.
    - iters (int): Maximum number of iterations for Dykstra's algorithm.
    - tol (float): Tolerance for convergence of Dykstra's algorithm.

    Returns:
    - rhoOut (array-like): Approximated density matrix.
    - finalTime - initTime (float): Time taken for the computation.
    - it (int): Number of iterations performed by Dykstra's algorithm.

    This function combines Dykstra's alternating projection algorithm with Cholesky-based approximation
    to approximate CPTP matrices. Dykstra's algorithm alternates between the density matrix set 
    via Smolin algorithm and the TP set. Then, Cholesky-based approximation is used to further refine 
    the approximation. The function returns the final approximated CPTP matrix, the computation
    time, and the number of iterations performed by Dykstra's algorithm.
    """    

    initTime = timeit.default_timer()
    pos_matrix, _, it = dykstra(matrix, "TPP", iters, tol)
    rhoOut, _ = cholesky_based_approximation(pos_matrix, pos = True)
    finalTime = timeit.default_timer()
    return rhoOut, finalTime - initTime,  it



def dykstra_identity(matrix, iters, tol): 
    """
    Perform Dykstra's alternating projection algorithm followed by identity-mixing correction.

    Args:
    - matrix (array-like): Input matrix.
    - iters (int): Maximum number of iterations for Dykstra's algorithm.
    - tol (float): Tolerance for convergence of Dykstra's algorithm.

    Returns:
    - rhoOut (array-like): Approximated density matrix.
    - computation_time (float): Time taken for the computation.
    - iterations (int): Number of iterations performed by Dykstra's algorithm.

    This function combines Dykstra's alternating projection algorithm with an identity-mixing 
    correction to approximate CPTP matrices. First, Dykstra's algorithm alternates between the PSD matrix set 
    and the TP set. Then,the identity mixing is used to further refine the approximation. 
    The function returns the final approximated density matrix, the computation time, 
    and the number of iterations performed by Dykstra's algorithm.
    """

    initTime = timeit.default_timer()
    tp_matrix, _, it = dykstra(matrix, "PTP", iters, tol)
    dim = tp_matrix.shape[0]
    evals, _ = np.linalg.eigh(tp_matrix)
    lambda_min = np.min(evals)
    if lambda_min < 0:
        p = lambda_min /(lambda_min - 1./dim)
        rhoOut = (1 - p) * tp_matrix + p * np.eye(dim) / dim
    else:
        rhoOut = tp_matrix
    finalTime = timeit.default_timer()
    return rhoOut, finalTime - initTime,  it


def HIP(matrix, Id=True, tol=1e-7):
    """
    Perform Hyperplane Intersection Projection (HIP) followed by an optional identity-mixing correction.

    Args:
    - matrix (array-like): Input matrix.
    - Id (bool): Flag to indicate whether to apply the identity-mixing correction. (default is True).
    - tol (float): Tolerance for convergence of HIP algorithm.

    Returns:
    - rho (array-like): Approximated density matrix.
    - finalTime - initTime (float): Time taken for the computation.

    This function performs Hyperplane Intersection Projection (HIP) to approximate CPTP
    matrices. Optionally, it applies an identity-mixing correction to ensure trace-preserving condition.
    The function returns the final approximated density matrix and the computation time.
    """

    initTime = timeit.default_timer()
    dim = matrix.shape[0]
    rho = hyperplane_intersection_projection_switch(matrix, least_ev_x_dim2_tol=tol)
    if Id:
        evals, _ = np.linalg.eigh(rho)
        lambda_min = np.min(evals)
        if lambda_min < 0:
            p = lambda_min /(lambda_min - 1./dim)
            rho = (1 - p) * rho + p * np.eye(dim) / dim
        finalTime = timeit.default_timer()
        return rho, finalTime - initTime
    else:
        rho, _ = cholesky_based_approximation(rho, False)
        finalTime = timeit.default_timer()
        return rho, finalTime - initTime
        
    

def ensure_trace(eigvals):
    """
    Assumes sum of eigvals is at least one.

    Finds the value l so that $\sum (\lambda_i - l)_+ = 1$
    and set the eigenvalues $\lambda_i$ to $(\lambda_i - l)_+$.
    """
    trace = eigvals.sum()
    while trace > 1:
        indices_positifs = eigvals.nonzero()
        l = len(indices_positifs[0]) # Number of (still) nonzero eigenvalues
        eigvals[indices_positifs] += (1 - trace) / l  
        eigvals = eigvals.clip(0)
        trace = eigvals.sum() 
    return eigvals
        


def proj_CP_threshold(rho,  free_trace=True, full_output=False, thres_least_ev=False):
    """
    If thres_least_ev=False and free_trace=False, then projects rho on CP
    trace_one operators.
    
    More generally, changes the eigenvalues without changing the eigenvectors:
    * if free_trace=True and thres_least_ev=False, then projects on CP operators,
    with no trace condition.
    * if thres_least_ev=True, free_trace is ignored. Then we bound from below all 
    eigenvalues by their original value plus the least eigenvalue (which is negative).
    Then all the lower eigenvalues take the lower bound (or zero if it is negative),
    all the higher eigenvalues are unchanged, and there is one eigenvalue in the middle
    that gets a value between its lower bound and its original value, to ensure the
    trace is one.
    """
    eigvals, eigvecs = sp.linalg.eigh(rho) # Assumes hermitian; sorted from lambda_min to lambda_max
    
    least_ev = eigvals[0]
    
    if thres_least_ev:
        threshold = - least_ev # > 0
        evlow = (eigvals - threshold).clip(0)
        toadd = eigvals - evlow
        missing = 1 - evlow.sum()
        if missing < 0: # On this rare event, revert to usual projection
            eigvals = eigvals.clip(0)
            eigvals = ensure_trace(eigvals)
        else:
            inv_cum_toadd =  toadd[::-1].cumsum()[::-1]
            last_more_missing = np.where(inv_cum_toadd >= missing)[0][-1]
            eigvals[:last_more_missing] = evlow[:last_more_missing]
            eigvals[last_more_missing] = eigvals[last_more_missing] + missing - inv_cum_toadd[last_more_missing]    
    else:
        eigvals = eigvals.clip(0)
        if not free_trace:
            eigvals = ensure_trace(eigvals)
        #    
    indices_positifs = eigvals.nonzero()[0]    
    rho_hat_TLS = (eigvecs[:,indices_positifs] * eigvals[indices_positifs]) @ eigvecs[:,indices_positifs].T.conj()
    
    if full_output==2:
        return rho_hat_TLS, least_ev, len(indices_positifs)
    elif full_output:
        return rho_hat_TLS, least_ev
    else:
        return rho_hat_TLS
        

def step2(XW, target):
    """
    Finds a (big) subset of hyperplanes, including the last one, such that
    the projection of the current point on the intersection of the corresponding
    half-spaces is the projection on the intersection of hyperplanes.

    Input: XW is the matrix of the scalar products between the different 
    non-normalized normal directions projected on the subspace TP, written w_i
    in the main functions.
    target is the intercept of the hyperplanes with respect to the starting point,
    on the scale given by w_i.

    Outputs which hyperplanes are kept in subset, and the coefficients on their
    respective w_i in coeffs.
    """
    nb_active = XW.shape[0]
    subset = np.array([nb_active - 1])
    coeffs = [target[-1] / XW[-1, -1]] # Always positive
    for i in range(nb_active - 2, -1, -1):
        test = (XW[i, subset].dot(coeffs) < target[i])
        # The condition to project on the intersection of the hyperplanes is that 
        # all the coefficients are non-negative. This is equivalent to belonging
        # to the normal cone to the facet.
        if test:
            subset = np.r_[i, subset]
            coeffs = sp.linalg.inv(XW[np.ix_(subset, subset)]).dot(target[subset]) 
            # Adding a new hyperplane might generate negative coefficients.
            # We remove the corresponding hyperplanes, except if it is the last 
            # hyperplane, in which case we do not add the hyperplane.
            if coeffs[-1] < 0: 
                subset = subset[1:]
                coeffs = sp.linalg.inv(XW[np.ix_(subset, subset)]).dot(target[subset]) 
            elif not np.all(coeffs >= 0):
                subset = subset[np.where(coeffs >= 0)]
                coeffs = sp.linalg.inv(XW[np.ix_(subset, subset)]).dot(target[subset])
    
    return subset, coeffs


def step_generator(x):
    """
    Yields an iterator from a number, tuple or list, looping eternally.
    If input is already an iterator, yields it unchanged.
    """
    if np.isscalar(x):
        def _gen(x):
            while True:
                yield x
        gen = _gen(x)
    elif isinstance(x, list) or isinstance(x, tuple):
        x = list(x)
        def _gen(x):
            i = 0
            ll = len(x)
            while True:
                yield x[i % ll]
                i+=1
        gen = _gen(x)
    elif isinstance(x, collections.Iterator):
        gen = x
    return gen


def hyperplane_intersection_projection_switch(rho, maxiter=100, free_trace=True,
                    least_ev_x_dim2_tol=1e-7, all_dists=False, dist_L2=True, with_evs=False, 
                    save_intermediate=False, HIP_to_alt_switch='first',
                    alt_to_HIP_switch='counter', min_cos = .99,
                    alt_steps=4, missing_w=1, min_part=.3, HIP_steps=10, 
                    max_mem_w=30, **kwargs):
    """ Switches between alternate projections and HIP, with the following rules:
    * starts in alternate projections.
    * stays in alternate depending on alt_to_HIP_switch:
        ** if 'counter': uses an iterator (alt_steps) of the iteration number to determine the 
        number of consecutive steps before switching. If alt_steps
        is a number, yields this number. If a list cycles on the list.
        ** if 'cos':  switching when two
        successive steps are sufficiently colinear, namely if the cosinus of
        the vectors is at least min_cos.
    * stays in HIP depending on HIP_to_alt_switch:
        ** if 'first': stops HIP when the first active hyperplane
        of the sequence gets discarded. (ex: enter at iteration 7, then leaves when 
        the hyperplane of iteration 7 is not in w_act anymore).
        ** if 'missing', stops when a total of missing_w (default 1) hyperplanes are 
        deemed unnecessary. (ie w_act has lost missing_w member).
        ** if 'part': ends the loop if the length coeff_first * w_first is less than min_part 
        times the step size, ie the length of \sum coeffs_i w_i. This includes the case when
        the first hyperplane is deemed unnecessary, like in 'first'.
        ** if 'counter': uses an iterator (HIP_steps) of the iteration number to determine the 
        number of consecutive steps before switching. Iterator in input iter_choice. If 
        HIP_steps is a number, yields this number. If a list cycles on the list.
    """
    
    dim2 = len(rho) 
    comp_time=0
    # x_sq, xiwi = -1, 1 # For the first entry in the loop. Yields the impossible -1.
    sel = 'alternate' # Selector for the step; 'alternate' or 'HIP'.
    if alt_to_HIP_switch == 'cos':
        w_norm_ancien = np.zeros((dim2, dim2)) # Not normalized to ensure at least two steps are taken.
    elif alt_to_HIP_switch == 'counter':
        past_al = 0       # number of steps already made in 'alternate' mode.
        alt_step_gen = step_generator(alt_steps)
        current_alt_step = next(alt_step_gen)
    else:
        raise ValueError('Unknown alt_to_HIP_switch. Must be "cos" or "counter".')

    if HIP_to_alt_switch == 'counter':
        HIP_step_gen = step_generator(HIP_steps)
        past_HIP = 0
    elif HIP_to_alt_switch == 'part':
        pass
    elif HIP_to_alt_switch == 'first':
        pass
    elif HIP_to_alt_switch == 'missing':
        missed = 0    
    else:
        raise ValueError('Unknown HIP_to_alt_switch. Must be "first", "missing", "part" or "counter".')



    dims = (dim2, dim2)

    active = np.array([])
    nb_actives = 0
    XW = np.zeros((0,0))
    w_act = np.zeros([0, dim2, dim2])
    target = np.array([])
    coeffs = np.array([])
    

    # rho is on CP, we first project on TP. Outside the loop because we also end on TP.
    t0 = time.perf_counter()
    rho = proj_tp(rho)
    t1 = time.perf_counter()
    

    for m in range(maxiter):
        
        #print(f'Enters iteration {m}')
        comp_time += t1 - t0
            
        # On CP
        t0 = time.perf_counter()
        rho_after_CP, least_ev = proj_CP_threshold(rho, free_trace, full_output=True)
        t1 = time.perf_counter()
        
        
        # Breaks here because the (- least_ev) might increase on the next rho
        if  (- least_ev) < least_ev_x_dim2_tol / dim2:
            t1 = t0 # Do not count twice the calculation time
            break
            
        t0 = time.perf_counter()       
        if (sel == 'alternate') or (m>=(maxiter-2)): # Ensures last ones are AP.
            #print('Alternate projections mode')

            # On TP and intersection with hyperplane
            if alt_to_HIP_switch == 'cos':
                w_new = proj_tp(rho_after_CP) - rho
                norm_w = sp.linalg.norm(w_new)
                change = (np.vdot(w_new / norm_w, w_norm_ancien).real > min_cos)
                w_norm_ancien = w_new / norm_w

                # If change with alt_steps, the current projection is transformed into
                # the first HIP step.
                if change:
                    active = np.array([m])
                    nb_actives = 1
                    XW = np.array([[norm_w**2]])
                    w_act = np.array([w_new])
                    coeffs = np.array([sp.linalg.norm(rho - rho_after_CP)**2 / norm_w**2])
                    target = np.array([0.])
                    rho += coeffs[0] * w_new
                    
                else:
                    rho += w_new
                    
            elif alt_to_HIP_switch == 'counter':
                rho = proj_tp(rho_after_CP)
                past_al += 1
                change = (past_al >= current_alt_step)

                if change:
                    active = np.array([])
                    nb_actives = 0
                    XW = np.zeros((0,0))
                    w_act = np.zeros([0, dim2, dim2])
                    target = np.array([])
                    coeffs = np.array([])

            if change:
                if HIP_to_alt_switch == 'missing':
                    missed = 0
                elif HIP_to_alt_switch == 'counter':
                    past_HIP = 0
                    current_HIP_step = next(HIP_step_gen)
                sel = 'HIP'

            t1 = time.perf_counter()

        elif sel == 'HIP': # No other possibility
            #print(f'HIP mode. Active hyperplanes: {1 + nb_actives}')

            sq_norm_x_i = sp.linalg.norm(rho - rho_after_CP)**2
            w_i =  proj_tp(rho_after_CP) - rho
            xiwi = sp.linalg.norm(w_i)**2
            
            XW = np.column_stack([XW, np.zeros(nb_actives)])
            XW = np.row_stack([XW, np.zeros(nb_actives + 1)])
            new_xw = np.einsum('ij, kij -> k', w_i.conj(), w_act).real # Notice that the scalar product are all real
                                                                      # since the matrices are self-adjoint.
            XW[-1, :-1] = new_xw
            XW[:-1, -1] = new_xw
            XW[-1, -1]  = xiwi
            target = np.r_[target, sq_norm_x_i]    
            
        
            active = np.concatenate((active, [m]))
            w_act = np.concatenate([w_act, [w_i]])

            subset, coeffs = step2(XW, target) 
            
            if HIP_to_alt_switch == 'missing':
                missed += len(active) - len(subset) # Don't move this after the update to active !!!
                                 
            XW = XW[np.ix_(subset, subset)]
            active = active[subset]
            nb_actives = len(active)
            w_act = w_act[subset]
            target = np.zeros((nb_actives,))
            rho += np.einsum('k, kij -> ij', coeffs, w_act)
   


            if HIP_to_alt_switch in ['first', 'part']:
                if (subset[0] != 0) or nb_actives > max_mem_w: # max_mem_w limits memory usage
                    change = True
                elif HIP_to_alt_switch == 'part':
                    step_size = np.sqrt(np.einsum('i, ij, j', coeffs, XW, coeffs))
                    w_first_contrib = coeffs[0] * np.sqrt(XW[0,0])
                    change = (min_part * step_size >= w_first_contrib)
                else:
                    change = False
            elif  HIP_to_alt_switch in ['counter', 'missing']:
                
                # Limits memory usage
                if nb_actives > max_mem_w:
                    nb_actives -= 1
                    active = active[1:]
                    w_act = w_act[1:]
                    target = target[1:]
                    XW = XW[1:, 1:]
                    if HIP_to_alt_switch == 'missing':
                        missed += 1
                # End max_mem_w case
                
                if HIP_to_alt_switch == 'missing':
                    change = (missed >= missing_w)
                elif HIP_to_alt_switch == 'counter':
                    past_HIP += 1
                    change = (past_HIP >= current_HIP_step)    

            if change:
                if alt_to_HIP_switch == 'cos':
                    w_norm_ancien = np.zeros((dim2, dim2)) # Ensures two alternate steps. Also possible to
                                                          # use w_norm_ancien = w_i / np.sqrt(xiwi)
                elif alt_to_HIP_switch == 'counter':
                    past_al = 0
                    current_alt_step = next(alt_step_gen)
                sel = 'alternate'

            t1 = time.perf_counter()

        else:
            raise ValueError('How did I get there? Typo on "HIP" or "alternate"?')
        
    return rho
    

def is_positive_semidefinite(matrix):
    """
    Check if a matrix is positive semidefinite.

    Args:
    - matrix (array-like): Input matrix to be checked.

    Returns:
    - bool: True if the matrix is positive semidefinite, False otherwise.

    This function calculates the eigenvalues of the input matrix and checks if all eigenvalues are
    non-negative. If all eigenvalues are greater than or equal to a small negative threshold (-1e-7),
    the function returns True, indicating that the matrix is positive semidefinite.
    """

    eigenvalues = np.linalg.eigvals(matrix)
    if np.all(eigenvalues >= -1e-7):
        return True
    else:
        return False
    

def is_trace_preserving(matrix):
    """
    Check if a matrix is trace preserving.

    Args:
    - matrix (array-like): Input matrix to be checked.

    Returns:
    - bool: True if the matrix is trace preserving, False otherwise.

    This function checks if the input matrix is trace-preserving by computing the partial trace over
    one subsystem and comparing it with the maximally mixed state of the same subsystem. If the partial
    trace is close to the maximally mixed state within a small threshold, the function returns True,
    indicating that the matrix is trace preserving.
    """

    dim, _ = matrix.shape
    d = int(np.sqrt(dim))
    qubits = int(np.log2(dim))
    rhoRes = matrix.reshape([d, d, d, d])
    rhoA = np.einsum('ijik->jk', rhoRes) # partial trace 
    var = np.isclose(rhoA, np.eye(int(2 ** (qubits/2)))/d, 1e-7)
    return np.all(var)
