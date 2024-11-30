import cvxopt
import numpy as np

########### TO-DO ###########
# 1. Implement linear kernel
#   --> See: def linear_kernel(x1, x2):
# 2. Implement rbf kernel
#   --> See: def rbf_kernel(x1, x2):
# 3. Implement fit
#   --> See: def fit(self, X, y):
#   --> Add matrix Q, p, G, h, A, b and save the solution
# 4. Implement predict
#   --> See: def predict(self, X):


class SVM:
    """Implements the support vector machine"""

    def __init__(self, kernel="linear", sigma=0.25):
        """Initialize perceptron."""
        self.__alphas = None
        self.__targets = None
        self.__training_X = None
        self.__bias = None
        if kernel == "linear":
            self.__kernel = SVM.linear_kernel
        elif kernel == "rbf":
            self.__kernel = SVM.rbf_kernel
            self.__sigma = sigma
        else:
            raise ValueError("Invalid kernel")

    @staticmethod
    def linear_kernel(x1, x2):
        """
        Computes the linear kernel between two sets of vectors.

        Args:
            x1 (numpy.ndarray): A matrix of shape (n_samples_1, n_features) representing the first set of vectors.
            x2 (numpy.ndarray): A matrix of shape (n_samples_2, n_features) representing the second set of vectors.

        Returns:
            numpy.ndarray: A matrix of shape (n_samples_1, n_samples_2) representing the linear kernel between x1 and x2.
        """
        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        # The linear kernel is simply the dot product of the two vectors
        return np.dot(x1, x2.T) 
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    @staticmethod
    def rbf_kernel(x1, x2, sigma):
        """
        Computes the radial basis function (RBF) kernel between two sets of vectors.

        Args:
            x1: A matrix of shape (n_samples_1, n_features) representing the first set of vectors.
            x2: A matrix of shape (n_samples_2, n_features) representing the second set of vectors.

        Returns:
            A matrix of shape (n_samples_1, n_samples_2) representing the RBF kernel between x1 and x2.
        """

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        # The RBF kernel is defined as exp(-||x1 - x2||^2 / (2 * sigma^2))
        # Here, we also make use of the identity ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2 * x1 * x2
        x1_sq = np.sum(x1**2, axis=1).reshape(-1, 1)
        x2_sq = np.sum(x2**2, axis=1).reshape(1, -1)
        return np.exp(-((x1_sq + x2_sq - 2 * np.dot(x1, x2.T)) / (2 * sigma**2)))
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    def fit(self, X, y):
        """Training function.

        Args:
            X (numpy.ndarray): Inputs.
            y (numpy.ndarray): labels/target.

        Returns:
            None
        """
        # n_observations -> number of training examples
        # m_features -> number of features
        n_observations, m_features = X.shape
        self.__norm = max(np.linalg.norm(X, axis=1))
        X = X / self.__norm
        y = y.reshape((1, n_observations))

        # quadprog and cvx all want 64 bits
        X = X.astype(np.float64)
        y = y.astype(np.float64)

        print("Computing kernel matrix...")
        if self.__kernel == SVM.linear_kernel:
            K = self.__kernel(X, X)
        elif self.__kernel == SVM.rbf_kernel:
            K = self.__kernel(X, X, self.__sigma)
        print("Done.")

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        # Minimize w.r.t. w and maximize w.r.t. alpha

        # The quadratic term encodes the dependency of the decision boundary on both the labels (y) and the kernel similarity between points K(x1, x2).
        Q = cvxopt.matrix(np.outer(y, y) * K)

        # The linear term encourages maximizing the sum of alpha, as seen in the dual objective.
        p = cvxopt.matrix(-np.ones(n_observations))

        # enforce the inequality alpha >= 0
        G = cvxopt.matrix(-np.eye(n_observations))
        h = cvxopt.matrix(np.zeros(n_observations))

        # enforce the equality constraint sum(alpha * y) = 0 through A * alpha = b
        A = cvxopt.matrix(y)
        b = cvxopt.matrix(0.0)
        # SEE: https://cvxopt.org/examples/tutorial/qp.html and https://cvxopt.org/userguide/coneprog.html#quadratic-programming and http://www.seas.ucla.edu/~vandenbe/publications/mlbook.pdf
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        cvxopt.solvers.options["show_progress"] = False
        solution = cvxopt.solvers.qp(Q, p, G, h, A, b)

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        # Extract alphas from solution
        self.__alphas = np.ravel(solution['x'])

        # Extract support vectors from labels
        self.__support_vector_labels = y[0]

        # Update bias based on the calculation of w0 in the slides
        self.__bias = np.mean(self.__support_vector_labels - np.dot(K, self.__alphas * self.__support_vector_labels))

        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        self.__targets = y
        self.__training_X = X

    def predict(self, X):
        """Prediction function.

        Args:
            X (numpy.ndarray): Inputs.

        Returns:
            Class label of X
        """

        X = X / self.__norm

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        # Compute the kernel matrix between the input data and the training data
        if self.__kernel == SVM.linear_kernel:
            K = self.__kernel(X, self.__training_X)
        elif self.__kernel == SVM.rbf_kernel:
            K = self.__kernel(X, self.__training_X, self.__sigma)

        # Construct decision function according to slides
        alphas_targets = self.__alphas * self.__targets
        decision_function = K @ alphas_targets.T + self.__bias

        # Return -1 if smaller than 0 and 1 if greater than 0
        return np.sign(decision_function)
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****
