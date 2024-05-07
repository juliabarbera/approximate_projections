import numpy as np 
import qutip as qt
import scipy
import matplotlib.pyplot as plt
import cvxpy as cp 
import time 
import timeit

class closestPOVM():
    def __init__(self, qubits, iters):
        super(closestPOVM, self).__init__()
        
        self.qubits = qubits
        self.dim = 2 ** self.qubits 
        self.iters = iters

    def pauli_bases(self):
        """
        Generate 3**n_qubits bases that define a IC POVM for quantum tomography.

        Args:
        - n_qubits (int): Number of qubits of the system.

        Returns:
        - bases (array-like): Array containing the Pauli bases.

        This method generates the Pauli bases for the quantum system represented by the qubits.
        It defines the Pauli matrices X, Y, and Z for each qubit and calculates all possible
        combinations of tensor products of these matrices to form the Pauli bases for the
        entire system. The result is returned as an array containing the Pauli bases.
        """
        
        # define the one-qubit bases
        X = np.array([[0., 1.], [1., 0.]], dtype=complex)
        Y = np.array([[0, -1.*1j], [1.*1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        sigma = np.stack([X, Y, Z], axis = -1)
        bases = np.zeros((2 ** self.qubits, 2 ** self.qubits, 3 ** self.qubits), dtype = "complex")
        # calculate all the combinations of tensor products
        for i in range(3 ** self.qubits):
            j = np.base_repr(i, base=3)
            j = "0" * (self.qubits - len(j)) + j
            basis_i = 1
            for k in range(self.qubits):
                basis_i = np.kron(basis_i, sigma[:, :, int(j[k])])
            bases[:, :, i] = basis_i

        return bases


    def base2operator(self, vec):
        """
        Given a list of bases, constructs the vector representations of their projectors.

        Args:
        - vec (bool): Whether to reshape the output to a vector (optional, default=False).

        Returns:
        - A (array-like): Array containing all observables.

        This method converts the Pauli bases to vectors. It selects a specific Pauli basis
        from the generated Pauli bases and calculates the corresponding vectorial representation by taking the
        outer product of the basis vectors. If 'vec' is True, the output is reshaped to a vector.
        """

        matrix = self.pauli_bases()[:, :, 10]
        dim = matrix.shape
        if len(dim) == 2:
            A = np.zeros((dim[0], dim[0], dim[1]), dtype="complex")
            for k in range(dim[1]):
                evector = matrix[:, k].reshape(-1, 1)
                projector = np.kron(evector.conj().T, evector)
                A[:, :, k] = projector
        else:
            A = np.zeros((dim[0], dim[0], dim[2]*dim[1]), dtype="complex")
            for j in range(dim[2]):
                for k in range(dim[1]):
                    evector = matrix[:, k, j].reshape(-1, 1)
                    projector = np.kron(evector.conj().T, evector)
                    A[:, :, dim[0]*j + k] = projector
        if vec:
            A = A.reshape(dim[0]**2, -1, order="F")
        return A


    def noisy_POVM(self, p):
        """
        Generate noisy POVM operators.

        Args:
        - p (float): Noise parameter.

        Returns:
        - ops (array-like): Array containing the noisy POVM operators.

        This method generates noisy POVM operators by adding random noise to the original
        operators and normalizing them.
        """

        X = self.base2operator(vec = False)
        di, _, N = X.shape
        ops = np.zeros((di, di, N), dtype=complex)
        for n in range(N):
            M = np.random.randn(di, di) + 1j*np.random.randn(di, di) 
            H = M + M.conj().T
            ops[:, :, n] = (1-p) * X[:, :, n]/np.trace(X[:, :, n]) + p*H/np.trace(H)
        return ops
        
        
    def proj_pos(self, matrices):
        """
        Project matrices onto the positive semidefinite cone.

        Args:
        - matrices (array-like): Array containing the input matrices.

        Returns:
        - pos_matrices (array-like): Array containing the projected positive semidefinite matrices.
        """

        di, _, N = matrices.shape
        pos_matrices = np.zeros((di, di, N), dtype=complex)
        for j in range(N):
            B = (matrices[:, :, j] + matrices[:, :, j].conj().T) / 2
            _, s, V = np.linalg.svd(B)
            H = np.dot(V.conj().T, np.dot(np.diag(s), V))
            A2 = (B + H) / 2
            A3 = (A2 + A2.conj().T) / 2
            pos_matrices[:, :, j] = A3
        return pos_matrices


    def proj_id(self, X):
        """
        Project matrices onto the set of matrices summing up to the identity.

        Args:
        - X (array-like): Array containing the input matrices.

        Returns:
        - Y (array-like): Array containing the projected matrices.
        """

        di, _, N = X.shape
        Y = X - np.sum(X, axis=2, keepdims=True)/N + np.eye(di, di).reshape(di, di, 1)/N
        return Y


    def split_pos_neg(self, X):
        """
        Split matrices into positive and negative parts based on the eigenvalues.

        Args:
        - X (array-like): Array containing the input matrices.

        Returns:
        - ops_p (array-like): Array containing the positive part of the input matrices.
        - ops_n (array-like): Array containing the negative part of the input matrices.
        """

        di, _, N = X.shape
        ops_p = np.zeros((di, di, N), dtype=complex)
        ops_n = np.zeros((di, di, N), dtype=complex)
        for j in range(N):
            evals, evecs = np.linalg.eigh(X[:, :, j])
            for k in range(di):
                evector = evecs[:, k:k+1]
                if evals[k] > 0:
                    ops_p[:, :, j] += evals[k]*np.kron(evector.conj().T, evector)
                else:
                    ops_n[:, :, j] += -evals[k]*np.kron(evector.conj().T, evector)
        return ops_p, ops_n
    

    def two_step_estimation(self, povmIn, Id): 
        """
        Perform two-step estimation algorithm.

        Args:
        - povmIn (array-like): Array containing the input POVM.
        - Id (bool): Whether to use identity projection (optional, default=True).

        Returns:
        - (finalTime - initTime) (float): Time taken for the computation.
        - povmOut (array-like): Array containing the estimated POVM.
        """ 

        # paper chinos
        initTime = timeit.default_timer()
        if Id == True: 
            Y_3 = povmIn
        elif Id == False:
            Y_3 = self.proj_id(povmIn)
        Y_3p, Y_3n = self.split_pos_neg(Y_3)
        C = scipy.linalg.cholesky(np.sum(Y_3n, axis=-1) + np.eye(self.dim), lower=True)
        U = scipy.linalg.sqrtm(C.conj().T @ C) @ scipy.linalg.inv(C)
        Y_3f = np.zeros(povmIn.shape, dtype=complex)
        for i in range(self.dim):
            Y_3f[:, :, i] = U.conj().T @ scipy.linalg.inv(C) @ Y_3p[:, :, i] @ scipy.linalg.inv(C.conj().T) @ U
        finalTime =  timeit.default_timer()
        return (finalTime - initTime), Y_3f


    def dykstra(self, Z, order, tol):
        """
        Perform Dykstra's alternating projection algorithm to approximate a valid POVM.

        Args:
        - Z (array-like): Input noisy POVM.
        - order (str): Order of projection (either "IP" or "PI").
        - it (int): Maximum number of iterations.
        - tol (float): Tolerance for convergence.

        Returns:
        - X_1 (array-like): Approximated POVM.
        - finalTime - initTime (float): Time taken for the computation.
        - it (int): Number of iterations performed.

        This function implements Dykstra's alternating projection algorithm to approximate a set of matrices that 
        define a POVM. It alternates between projecting onto the set of positive semidefinite matrices
        and the set of adding to the identity. The algorithm terminates when either the maximum
        number of iterations is reached or the change in the projection variables is below the specified tolerance.
        The function returns the approximated POVM, the computation time, and the number
        of iterations performed.
        """

        initTime = timeit.default_timer()
        X_1 = Z
        q = np.zeros((self.dim, self.dim, self.dim), dtype=complex)
        p = np.zeros((self.dim, self.dim, self.dim), dtype=complex)
        it = 0
        cI = 100
        if order == "IP":
            p_prev = p.copy()
            q_prev = q.copy()
            while it <= self.iters and cI >= tol:
                Y_1 = self.proj_id(X_1 + p)
                p = X_1 + p - Y_1
                X_1 = self.proj_pos(Y_1 + q)
                q = Y_1 + q - X_1
                cI = np.linalg.norm(p[:, :, 1] - p_prev[:, :, 1])**2 + np.linalg.norm(q[:, :, 1] - q_prev[:, :, 1])**2
                p_prev = p.copy()
                q_prev = q.copy()
                it += 1
            finalTime = timeit.default_timer()
            return X_1, finalTime - initTime, it 
        elif order == "PI":
            p_prev = p.copy()
            q_prev = q.copy()
            while it <= self.iters and cI >= tol:
                Y_1 = self.proj_pos(X_1 + p)
                p = X_1 + p - Y_1
                X_1 = self.proj_id(Y_1 + q)
                q = Y_1 + q - X_1
                cI = np.linalg.norm(p[:, :, 1] - p_prev[:, :, 1])**2 + np.linalg.norm(q[:, :, 1] - q_prev[:, :, 1])**2
                p_prev = p.copy()
                q_prev = q.copy()
                it += 1
            finalTime = timeit.default_timer()
            return X_1, finalTime - initTime, it 


    def cholesky_based_approximation(self, povmIn, pos): 
        """
        Perform Cholesky-based approximation to obtain a valid POVM.

        Args:
        - povmIn (array-like): Input povm.
        - pos (bool): Flag to indicate whether to enforce positivity.

        Returns:
        - finalTime - initTime (float): Time taken for the optimization process.
        - rhoOut (array-like): positive and summing to identity POVM.

        This function implements the Cholesky-based approximation.
        In the first stage, if specified, it projects the input set of matrices onto the set of positive 
        matrices. Next, it projects the matrix onto the set of summing to identity matrices
        via unitary transformations.
        """  
        initTime = timeit.default_timer()
        if pos == True: 
            pass
        elif pos == False:  
            povmIn = self.proj_pos(povmIn)
        i_1_minus_lambda = scipy.linalg.sqrtm(sp.linalg.inv(np.sum(povmIn, axis=-1)))
        povmOut = np.zeros(povmIn.shape, dtype=complex)
        for i in range(self.dim):
            povmOut[:, :, i] = i_1_minus_lambda @ povmIn[:, :, i] @ i_1_minus_lambda.conj().T
        finalTime = timeit.default_timer()
        return (finalTime - initTime), povmOut
    

    def dykstra_cholesky_based_approximation(self, Z, tol): 
        """
        Perform Dykstra's alternating projection algorithm followed by Cholesky-based approximation.

        Args:
        - Z (array-like): Input matrix.
        - iters (int): Maximum number of iterations for Dykstra's algorithm.
        - tol (float): Tolerance for convergence of Dykstra's algorithm.

        Returns:
        - finalTime - initTime (float): Time taken for the computation.
        - povmOut (array-like): Approximated POVM.
        - it (int): Number of iterations performed by Dykstra's algorithm.

        This function combines Dykstra's alternating projection algorithm with Cholesky-based approximation
        to approximate POVMs. Dykstra's algorithm alternates between the postive semidefinite set 
        of matrices and the set of matrices that sum to the identity. Then, Cholesky-based approximation is used to further refine 
        the approximation. The function returns the final approximated POVM, the computation
        time, and the number of iterations performed by Dykstra's algorithm.
        """   

        initTime = timeit.default_timer()
        pos_povm, _, it = self.dykstra(Z, "IP", tol)
        _,  povmOut = self.cholesky_based_approximation(pos_povm, pos = True)
        finalTime = timeit.default_timer()
        return  (finalTime - initTime), povmOut, it


    def dykstra_two_step_approximation(self, Z, tol): 
        """
        Perform Dykstra's alternating projection algorithm followed by two-step approximation.

        Args:
        - Z (array-like): Input matrix.
        - iters (int): Maximum number of iterations for Dykstra's algorithm.
        - tol (float): Tolerance for convergence of Dykstra's algorithm.

        Returns:
        - finalTime - initTime (float): Time taken for the computation.
        - povmOut (array-like): Approximated POVM.
        - it (int): Number of iterations performed by Dykstra's algorithm.

        This function combines Dykstra's alternating projection algorithm with Cholesky-based approximation
        to approximate POVMs. Dykstra's algorithm alternates between the postive semidefinite set 
        of matrices and the set of matrices that sum to the identity. Then, two-step approximation is used to further refine 
        the approximation. The function returns the final approximated POVM, the computation
        time, and the number of iterations performed by Dykstra's algorithm.
        """ 

        initTime = timeit.default_timer()
        id_povm, _, it = self.dykstra(Z, "PI", tol)
        _, povmOut = self.two_step_estimation(id_povm, Id = True)
        finalTime = timeit.default_timer()
        return  (finalTime - initTime), povmOut, it 


    def exactSDP(self, Z):
        """ 
        Solve the exact QPT problem using Semidefinite Programming (SDP).

        Args:
        - Z (array-like): POVM to be reconstructed.

        Returns:
        - endTime - initTime (float): Time taken for the reconstruction process.
        - problem.compilation_time (float): Compilation time taken by the optimization.
        - effects (array-like): Reconstructed POVM.


        This function solves the exact QDT problem using Semidefinite Programming (SDP).
        It formulates the problem as an SDP by defining the povm 'Z' as a hermitian-valued
        variable and setting up appropriate constraints. The constraints enforce hermiticity, positivity,
        and adding to the identity of the reconstructed povm. The objective function minimizes the
        Frobenius norm of the difference between the reconstructed and target matrices.
        The SDP problem is then solved using the SCS solver with a specified tolerance.
        The function returns the reconstructed elements of the povm and the time taken for the reconstruction.
        """
        dim, _, num_outcomes = Z.shape
        initTime = time.time()
        Pi_list = {}  # list of effects that are variables in cvxpy
        for jj in range(num_outcomes):
            Pi_list[jj] = cp.Variable((dim, dim), hermitian=True)

        # the sum of the effects gives the identity
        constraints_list = [np.sum([Pi_list[jj] for jj in range(num_outcomes)]) == np.eye(dim)]
        for jj in range(0, num_outcomes):
            constraints_list.append(Pi_list[jj] >> 0)  # the effects are positive
            norm = 0  # norm we minimize
            for jj in range(0, num_outcomes):
                norm += cp.norm(Pi_list[jj] - Z[:, :, jj], "fro") ** 2
                obj = cp.Minimize(norm)

        problem = cp.Problem(obj, constraints_list)
        problem.solve(solver = "SCS", eps = 1e-8)
        endTime = time.time()
        effects = np.zeros((dim, dim, num_outcomes), dtype=complex)
        for jj in range(num_outcomes):
            effects[:, :, jj] = Pi_list[jj].value

        return  endTime - initTime, problem.compilation_time, effects
