import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

########### TO-DO ###########
# 1. Add square loss:
#   --> See: class SquareLoss(Loss)
# 2. Add activation function (SIGMOID)
#   --> See: self.activation_functions AND self.activ_derivative AND self.activation_function_threshold
#   --> tanh is given as example
# 3. Implement the forward and backward pass
#   --> See: def fit()


class SquareLoss(object):
    """
    Square loss function.
    """

    def __init__(self, threshold, activation):
        self.threshold = threshold
        self.activation = activation

    def loss(self, y_true, y_pred):
        """
        Computes the square loss between the true and predicted value.

        Args:
            y (numpy.ndarray): True value.
            y_pred (numpy.ndarray): Predicted value.

        Returns:
            numpy.ndarray: Square loss value.
        """

        y_true = y_true if self.activation == "sigmoid" else np.where(y_true > 0, 1, -1)

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        # Square loss L2 loss= (true - predicted value)^2
        return (y_true - y_pred) ** 2
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    def delta(self, y_true, y_pred):
        """
        Computes the derivative of the square loss function.

        Args:
            y (numpy.ndarray): True value.
            y_pred (numpy.ndarray): Predicted value.

        Returns:
            numpy.ndarray: Derivative of the square loss function.
        """

        y_true = y_true if self.activation == "sigmoid" else np.where(y_true > 0, 1, -1)

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        # Derivative of square loss = 2*(predicted-true value)
        return 2 * (y_pred - y_true)
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    def calculate_accuracy(self, y_true, y_pred):
        """
        Computes the accuracy of the model for a given prediction.

        Args:
            y_pred (numpy.ndarray): Predicted value.
            y (numpy.ndarray): True value.

        Returns:
            float: Accuracy value.
        """
        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        # Convert predictions and ground truth to binary labels based on the threshold
        pred_labels = (y_pred >= self.threshold).astype(int)
        true_labels = (y_true >= self.threshold).astype(int)
        #Compute the percentage of correct labels by averaging over all zeros for wrong and ones for correct labels
        return np.mean(pred_labels == true_labels)
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****


# Activation functions and their derivatives
activation_functions = {
            # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
            # use definition of sigmoid S-shaped curve 1/(1+e^-x) as activation function
            "sigmoid": lambda x: 1/(1+np.exp(-x)), 
            # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****
            "tanh": lambda x: np.tanh(x),
}

activation_function_threshold = {
            # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
            # use default threshold of sigmoid 
            "sigmoid": 0.5,
            # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****
            "tanh": 0,
}

activation_derivatives = {
            # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
            # compute dervative of activation function x for sigmoid 
            "sigmoid": lambda x: x*(1-x),
            # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****
            "tanh": lambda x: 1 - x**2,
}

class MultiLayerPerceptron:
    """
    Multi-layer perceptron class for binary classification.

    Args:
        input_dim (int): Input dimensions.
        hidden_dim (int): Hidden layer dimensions.
        hidden_dim2 (int): Second hidden layer dimensions.
        output_dim (int): Output dimensions.
        lr (float): Learning rate.
        epochs (int): Number of epochs.
        activation (str): Activation function.
        weight_init (str): Initialization mode.

    Attributes:
        activation_functions (dict): Dictionary of activation functions.
        activation_derivatives (dict): Dictionary of derivative functions.
        activation_threshold (dict): Dictionary of activation function thresholds.

        input_dims (int): Input dimensions.
        hidden_dims (int): Hidden layer dimensions.
        hidden_dims2 (int): Second hidden layer dimensions.
        output_dims (int): Output dimensions.

        lr (float): Learning rate.
        epochs (int): Number of epochs.

        activation_function (function): Activation function.
        derivative_function (function): Derivative function.
        threshold (float): Activation function threshold.

        hidden_weight (numpy.ndarray): First hidden layer weights.
        hidden_weight2 (numpy.ndarray): Second hidden layer weights.
        output_weight (numpy.ndarray): Output layer weights.
        hidden_bias (numpy.ndarray): First hidden layer bias.
        hidden_bias2 (numpy.ndarray): Second hidden layer bias.
        output_bias (numpy.ndarray): Output layer bias.

    Methods:
        init_weights: Initializes weights.
        backprop: Performs backpropagation.
        fit: Fits the model.
    """

    def __init__(
        self,
        input_dim=256,
        hidden_dim=32,
        hidden_dim2=16,
        output_dim=1,
        lr=0.005,
        epochs=100,
        activation="sigmoid",
        weight_init="xavier"
    ):
        # Neural network architecture and training parameters
        self.input_dims = input_dim
        self.hidden_dims = hidden_dim
        self.hidden_dims2 = hidden_dim2
        self.output_dims = output_dim

        self.lr = lr
        self.epochs = epochs

        self.activation_function = activation_functions[activation]
        self.derivative_function = activation_derivatives[activation]
        self.threshold = activation_function_threshold[activation]

        # Initializing weights and biases
        self.hidden_weight = self.init_weights(self.input_dims, self.hidden_dims, weight_init)
        self.hidden_weight2 = self.init_weights(self.hidden_dims, self.hidden_dims2, weight_init)
        self.output_weight = self.init_weights(self.hidden_dims2, self.output_dims, weight_init)

        self.hidden_bias = self.init_bias(
            (self.hidden_dims), weight_init
        ) 
        self.hidden_bias2 = self.init_bias(
            (self.hidden_dims2), weight_init
        ) 
        self.output_bias = self.init_bias(
            (self.output_dims), weight_init
        ) 

        # Loss function
        self.loss = SquareLoss(self.threshold, activation=activation)

        # Initialize arrays for tracking loss and epochs
        self.loss_train_plot = []
        self.loss_test_plot = []
        self.acc_train_plot = []
        self.acc_test_plot = []

    def init_bias(self, x, mode="xavier"):
        """
        Initializes the bias for a layer of the MLP.

        Args:
            x (int): The number of neurons in the layer.
            mode (str): The initialization mode to use. Can be "random", "xavier"
                random: Random values between 0 and 0.001.
                xavier: Random values between -1/sqrt(x) and 1/sqrt(x). (recommended, standard in PyTorch)
                zeros: All values are set to 0.

        Returns:
            bias (numpy.ndarray): The initialized bias vector.
        """
        if mode == "random":
            bias = np.random.random((x)).astype(np.float32) * 0.001
        elif mode == "xavier":
            limit = 1 / math.sqrt(x)
            bias = np.random.uniform(-limit, limit, (x)).astype(np.float32)
        elif mode == "zeros":
            bias = np.zeros((x)).astype(np.float32)
        else:
            raise ValueError(f"Invalid initialization mode: {mode}")
        return bias

    def init_weights(self, x, y, mode="xavier"):
        """
        Initializes the weights of the neural network layer.

        Args:
            x (int): The number of input neurons.
            y (int): The number of output neurons.
            mode (str): The initialization mode to use. Can be "random", "xavier"
                random: Random values between 0 and 0.001.
                xavier: Random values between -1/sqrt(x) and 1/sqrt(x). (recommended, standard in PyTorch)
                zeros: All values are set to 0.


        Returns:
            weight (ndarray): The initialized weights as a numpy array.
        """
        if mode == "random":
            weight = np.random.random((x, y)).astype(np.float32) * 0.001
        elif mode == "xavier":
            limit = 1 / math.sqrt(y)
            weight = np.random.uniform(-limit, limit, (x, y))
        elif mode == "zeros":
            weight = np.zeros((x, y))
        else:
            raise ValueError(f"Invalid initialization mode: {mode}")
        return weight

    def backward(self, input, pred, gt):
        """
        Performs backpropagation.

        Args:
            input (numpy.ndarray): Input values.
            pred (numpy.ndarray): Predicted values.
            gt (numpy.ndarray): Ground truth values.
        """

        """
        This code calculates the error and derivatives for each layer of a neural network.
        It then updates the weights and biases for each layer based on the calculated error and derivatives.

        Args:
            input (numpy.ndarray): Input values.
            pred (numpy.ndarray): Predicted values.
            gt (numpy.ndarray): Ground truth values.
        """

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        # 1. Calculate error and derivatives for the output layer (Layer 3)
        # Use the derivative of the loss function (delta) to compute the difference, 
        # between the predicted output (pred) and the ground truth labels (gt).
        output_delta = self.loss.delta(gt, pred)
        # Multiply the output of the second hidden layer by the output layer's error signal
        # to compute the gradient for the output layer weights.
        # This represents how much the output weights need to be adjusted.
        output_adjustment = np.outer(self.fc2_w_acti, output_delta)

        # 2. Calculate error and derivatives for the hidden layer 2 (Layer 2)
        # Backpropagate the error to the second hidden layer
        # Multiply the output error by the transposed output weights to calculate 
        # the error contribution from the output layer to the second hidden layer, 
        # multiply with derivative of activation to account for non-linearity
        hidden2_delta =  np.dot(output_delta, self.output_weight.T)*self.derivative_function(self.fc2_w_acti)
        hidden2_adjustment = np.outer(self.fc1_w_acti, hidden2_delta)

        # 3. Calculate error and derivatives for the hidden layer 1 (Layer 1)
        hidden1_delta = np.dot(hidden2_delta, self.hidden_weight2.T) * self.derivative_function(self.fc1_w_acti)
        hidden1_adjustment = np.outer(input, hidden1_delta)

        # 4. Update weights for each layer
        # Subtract the learning rate multiplied by the respective weight adjustments.
        # This reduces the weights in the direction of the negative gradient to minimize the loss.
        self.output_weight -= self.lr * output_adjustment
        self.hidden_weight2 -= self.lr * hidden2_adjustment
        self.hidden_weight -= self.lr * hidden1_adjustment

        # 5. Update biases for each layer
        # Subtract the learning rate multiplied by the sum of the respective error signals, using the tip from the lecture
        # This adjusts the biases to account for the aggregated error at each layer.
        self.output_bias -= self.lr * output_adjustment[0, :]
        self.hidden_bias2 -= self.lr * hidden2_adjustment[0, :]
        self.hidden_bias -= self.lr * hidden1_adjustment[0, :]

        '''
                # 5. Update biases for each layer
        self.output_bias -= self.lr * np.sum(output_adjustment, axis=0, keepdims=False)
        self.hidden_bias2 -= self.lr * np.sum(hidden2_adjustment, axis=0, keepdims=False)
        self.hidden_bias -= self.lr * np.sum(hidden1_adjustment, axis=0, keepdims=False)'''

        '''
        # 4. Update weights for each layer
        self.output_weight += self.output_weight  # to be corrected by you
        self.hidden_weight2 += self.hidden_weight2  # to be corrected by you
        self.hidden_weight += self.hidden_weight  # to be corrected by you

        # 5. Update biases for each layer
        self.output_bias += self.output_bias  # to be corrected by you
        self.hidden_bias2 += self.hidden_bias2  # to be corrected by you
        self.hidden_bias += self.hidden_bias  # to be corrected by you
        '''
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    def forward(self, inputs):
        """One forward pass through the network.
        Args:
            inputs (numpy.ndarray): Input values.

        Returns:
            numpy.ndarray: Output values.
        """

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        # 1. Pass through hidden fully-connected layer 1
        # Compute the input for the first layer as a linear transformation, 
        # multiplying it with the weights and adding the biases
        self.net1 = (inputs @ self.hidden_weight) + self.hidden_bias  # to be corrected by you
        # Applying the activation function to the liner combination, 
        # introducing non-linearity and producing the output of the first hidden layer
        self.fc1_w_acti = self.activation_function(self.net1)

        # 2. Pass through hidden fully-connected layer 2
        # Apply the linear transformation to the output of the first hidden layer 
        # with weights and biases of the second hidden layer
        self.net2 = (self.fc1_w_acti @ self.hidden_weight2) + self.hidden_bias2 # to be corrected by you
        # Apply activation function to the linear combination
        self.fc2_w_acti = self.activation_function(self.net2)

        # 3. Pass through output fully-connected layer
        # Apply the linear transformation to the output of the second hidden layer 
        # with weights and biases of the output layer
        self.net3 = (self.fc2_w_acti @ self.output_weight) + self.output_bias
        # returning the linear combination directly without applying 
        # the activation function, since this produces better classification results
        self.fc3_w_acti = self.net3

        '''
        print("==============INPUTS==============")
        print(inputs)
        print("==============WEIGHTS1==============")
        print(self.hidden_weight)
        print("==============NET1==============")
        print(self.net1)
        print("==============ACT1==============")
        print(self.fc1_w_acti)
        print("==============WEIGHTS2==============")
        print(self.hidden_weight2)
        print("==============NET2==============")
        print(self.net2)
        print("==============ACT2==============")
        print(self.fc2_w_acti)
        print("==============WEIGHTS3==============")
        print(self.output_weight)
        print("==============NET3==============")
        print(self.net3)
        print("==============ACT3============== (NOT USED)")
        print(self.activation_function(self.net3))
        print("==============ACT3_THRESH============== (USED)")
        print(self.fc3_w_acti)'''

        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        return self.fc3_w_acti

    def fit(self, X_train, y_train, X_test, y_test):
        """
        Fits the model.

        Args:
            X (numpy.ndarray): Input values.
            y (numpy.ndarray): Ground truth values.

        Returns:
            MultiLayerPerceptron: Fitted model.
        """
        n = len(X_train)

        y_train[y_train==-1] = 0
        y_test[y_test==-1] = 0

        # Initialize arrays for tracking loss and epochs

        self.loss_train_plot = []
        self.loss_test_plot = []
        self.acc_train_plot = []
        self.acc_test_plot = []

        # Normalize input data to a range of [0, 1]
        X_train = X_train / 255
        X_test = X_test / 255

        training_idx = np.arange(n)

        # Initialize progress bar
        pbar = trange(self.epochs)

        for current_epoch in pbar:
            epoch_loss_train = []
            epoch_acc_train = []
            epoch_loss_test = []
            epoch_acc_test = []

            np.random.shuffle(training_idx)

            # Train one epoch
            for idx in training_idx:
                # shuffle training data and labels

                # Stage 1 - Forward Propagation
                pred = self.forward(X_train[idx])

                epoch_loss_train.append(self.loss.loss(y_train[idx], pred))
                epoch_acc_train.append(self.loss.calculate_accuracy(y_train[idx], pred))

                # Stage 2 - Backpropagation to update weights
                self.backward(X_train[idx], pred, y_train[idx])

            # Evaluate the model on the test set
            for idx, inputs in enumerate(X_test):
                # Stage 1 - Forward Propagation
                pred = self.forward(inputs)

                epoch_loss_test.append(self.loss.loss(y_test[idx], pred))
                epoch_acc_test.append(self.loss.calculate_accuracy(y_test[idx], pred))

            # Update progress bar with the current epoch's error
            pbar.set_description(
                f"Epoch {current_epoch + 1} - Loss (Train) {np.mean(epoch_loss_train):.5f} - Loss (Test) {np.mean(epoch_loss_test):.5f} - Acc (Train) {np.mean(epoch_acc_train)*100:.5f} - Acc (Test) {np.mean(epoch_acc_test)*100:.5f}"
            )

            self.loss_train_plot.append(np.mean(epoch_loss_train))
            self.loss_test_plot.append(np.mean(epoch_loss_test))
            self.acc_train_plot.append(np.mean(epoch_acc_train))
            self.acc_test_plot.append(np.mean(epoch_acc_test))

