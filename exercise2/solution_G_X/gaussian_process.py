import numpy as np
import scipy
import scipy.optimize as opt
from scipy.linalg import cho_solve, cholesky, solve_triangular
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared

###################### TO-DO #######################
########### Gaussian Process (30 Points) ###########
# NOTE: You may use the imported functions from scipy,
#       but you can also use another library if you want,
#       or implement the functions yourself.
# 1. Implement sampling (unconditioned): (2 Points)
#   --> See: sample_points(observation=None)
#   --> You may ignore the conditioned parameter for now
# 2. Update sampling (conditioned): (3 Points)
#   --> See: sample_points(observation=x) # x could be any number (for example 1.2)
# 3. Implement prediction based only on prior (8 Points)
#   --> See: GaussianProcess.predict()
# 4. Implement prediction based on posterior (8 Points)
#   --> See: GaussianProcess.predict()
#   --> And: GaussianProcess.fit(meta_parameter_search=False)
#   --> You may ignore the if neg_log_likelihood for now
# 5. Implement negative log likelihood type 2 (9 Points)
#   --> See: GaussianProcess.negative_log_likelihood_type_2()
#   --> And: GaussianProcess.fit(meta_parameter_search=True)


def sample_points(mean, cov, n, observation=None):
    """
    The function generates n samples from a multivariate normal distribution
    specified by the given mean and covariance matrix. If the observation
    parameter is set to a number, it generates samples
    conditioned on the observed number of the first variable.

    Args:
        mean (numpy.ndarray): Mean of the distribution
        cov (numpy.ndarray): Covariance matrix of the distribution
        n (int): Number of points to sample
        observation (int): the number to condition the second variable on

    Returns:
        numpy.ndarray: Sampled points
    """

    # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
    if observation is not None:
        # Conditioned sampling
        # Split covariance matrix into relevant components
        cov_11 = cov[1, 1]  # Variance of the second variable
        cov_12 = cov[1, 0]  # Covariance between the first and second variable
        cov_22 = cov[0, 0]  # Variance of the first variable
        
        # Compute the conditional mean for the second variable
        mean_cond = mean[1] + cov_12 / cov_22 * (observation - mean[0])
        # Compute the conditional variance for the second variable
        var_cond = cov_11 - (cov_12**2) / cov_22

        # Sample from the conditional distribution for the second variable
        sampled_x2 = np.random.normal(mean_cond, np.sqrt(var_cond), n)
        # The first variable is fixed at the observed value
        sampled_x1 = np.full(n, observation)
    else:
        # Unconditioned sampling
        # Directly sample from the multivariate normal distribution
        samples = np.random.multivariate_normal(mean, cov, n)
        sampled_x1, sampled_x2 = samples[:, 0], samples[:, 1]

    # Combine sampled variables into a single 2D array (2, n)
    sampled_points = np.vstack((sampled_x1, sampled_x2))
    # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    return sampled_points


class MultivariateNormal:
    def __init__(self, mean, cov, seed=42):
        self.mean = mean
        self.covariance = cov
        self.seed = seed

        self.distr = scipy.stats.multivariate_normal(mean, cov, seed=seed)

    def pdf(self, X, Y):
        return self.distr.pdf(np.dstack((X, Y)))


class GaussianProcess:
    def __init__(self, length_scale=1.0, noise=1e-10, kernel=None, periodicity=1.0):
        # Hyperparameters
        self.length_scale = length_scale  # Hyperparameter for length scale
        self.periodicity = periodicity  # Hyperparameter for periodicity
        self.noise = noise  # Hyperparameter for noise

        # Training Data and Related Variables
        self.X_train = None  # Placeholder for training data (input features)
        self.y_train = None  # Corresponding labels for training data
        self.n_targets = None  # Number of targets or outputs

        # Kernel-related Variables
        self.K = None  # Kernel matrix
        self.alpha_ = None  # Alpha variable related to the kernel
        self.L_ = None  # Lower triangular matrix
        self.kernel = kernel  # Kernel function
        self.kernel_type = kernel  # Variable storing the type of kernel

        assert self.kernel_type in [
            "RBF",
            "RBF+Sine",
            "Sine+RBF",
            "Sine",
        ], "Invalid kernel type"

    def negative_log_likelihood_type_2(self, params):
        """
        Negative log likelihood function.

        Args:
            params (numpy.ndarray): The parameters to optimize

        Returns:
            float: The negative log likelihood
        """

        length_scale, noise, periodicity = params

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        length_scale, noise, periodicity = params

        # Define the kernel based on the provided hyperparameters
        if self.kernel_type in ["RBF+Sine", "Sine+RBF"]:
            kernel = RBF(length_scale=length_scale) + ExpSineSquared(length_scale=length_scale, periodicity=periodicity)
        elif self.kernel_type == "RBF":
            kernel = RBF(length_scale=length_scale)
        elif self.kernel_type == "Sine":
            kernel = ExpSineSquared(length_scale=length_scale, periodicity=periodicity)

        # Compute the kernel matrix and add noise for numerical stability
        K = kernel(self.X_train) + noise * np.eye(len(self.X_train))
        # Perform Cholesky decomposition for efficient matrix inversion
        L = cholesky(K, lower=True)
        # Solve the linear system to compute alpha
        alpha = cho_solve((L, True), self.y_train)

        # Compute the log likelihood using the standard Gaussian log likelihood formula
        log_likelihood = (
            -0.5 * np.dot(self.y_train, alpha)  # Data fit term
            - np.sum(np.log(np.diagonal(L)))    # Complexity penalty
            - len(self.X_train) / 2 * np.log(2 * np.pi)  # Normalization term
        )
        return -log_likelihood  # Return the negative log likelihood (to minimize)
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    def fit(self, X_train, y_train, meta_parameter_search=False):
        """
        Fit the Gaussian Process model to the training data.

        Parameters:
        - X_train: Input features for training (numpy array)
        - y_train: Target values for training (numpy array)
        """
        self.X_train = X_train
        self.y_train = y_train

        # Update hyperparameters
        if meta_parameter_search:
            print(
                f"Parameters before: Lengthscale: {self.length_scale}, Noise: {self.noise}, Periodicity: {self.periodicity}"
            )

            # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
            res = opt.minimize(
                self.negative_log_likelihood_type_2,
                x0=[self.length_scale, self.noise, self.periodicity],
                bounds=[(1e-2, 1e2), (1e-10, 1e-1), (1e-2, 1e2)]
            )
            # Update hyperparameters with the optimized values
            self.length_scale, self.noise, self.periodicity = res.x
            # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

            print(
                f"Parameters after: Lengthscale: {self.length_scale}, Noise: {self.noise}, Periodicity: {self.periodicity}"
            )

        if self.kernel_type == "RBF+Sine" or self.kernel_type == "Sine+RBF":
            self.kernel = RBF(length_scale=self.length_scale) + ExpSineSquared(
                length_scale=self.length_scale, periodicity=self.periodicity
            )
        elif self.kernel_type == "RBF":
            self.kernel = RBF(length_scale=self.length_scale)
        elif self.kernel_type == "Sine":
            self.kernel = ExpSineSquared(
                length_scale=self.length_scale, periodicity=self.periodicity
            )

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        # Compute the kernel matrix and add noise
        self.K = self.kernel(self.X_train) + self.noise * np.eye(len(self.X_train))
        # Perform Cholesky decomposition for efficient computations
        self.L_ = cholesky(self.K, lower=True)
        # Solve for alpha (used in predictions)
        self.alpha_ = cho_solve((self.L_, True), self.y_train)
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    def predict(self, X_test):
        """
        Make predictions on new data.

        Parameters:
        - X_test: Input features for prediction (numpy array)

        Returns:
        - mean: Predicted mean for each input point
        - std: Predicted standard deviation for each input point
        """

        if (
            not hasattr(self, "X_train") or self.X_train is None
        ):  # Unfitted;predict based on GP prior
            # If the GP is called unfitted, we need to set the kernel here
            if self.kernel_type == "RBF+Sine" or self.kernel_type == "Sine+RBF":
                self.kernel = RBF(length_scale=self.length_scale) + ExpSineSquared(
                    length_scale=self.length_scale, periodicity=self.periodicity
                )
            elif self.kernel_type == "RBF":
                self.kernel = RBF(length_scale=self.length_scale)
            elif self.kernel_type == "Sine":
                self.kernel = ExpSineSquared(
                    length_scale=self.length_scale, periodicity=self.periodicity
                )

            # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
            y_mean_noisy = np.zeros(X_test.shape[0])  # Mean of the prior
            y_std_noisy = np.ones(X_test.shape[0])    # Std. dev. of the prior
            y_cov_noisy = np.eye(X_test.shape[0])     # Covariance of the prior
            # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

            return y_mean_noisy, y_std_noisy, y_cov_noisy

        else:  # Predict based on GP posterior
            # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
            # Posterior predictions
            K_s = self.kernel(self.X_train, X_test) # Cross-covariance between train and test
            K_ss = self.kernel(X_test) + self.noise * np.eye(len(X_test)) # Test covariance
            v = solve_triangular(self.L_, K_s, lower=True) # Solve triangular system

            mean_pred_distribution = np.dot(v.T, self.alpha_) # Compute posterior mean
            std_pred_distribution = np.sqrt(np.diag(K_ss - np.dot(v.T, v))) # Compute posterior variance
            conv_pred_distribution = K_ss - np.dot(v.T, v) # Compute posterior covariance
            # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

            return mean_pred_distribution, std_pred_distribution, conv_pred_distribution
