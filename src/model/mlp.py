
import numpy as np

from util.loss_functions import CrossEntropyError, BinaryCrossEntropyError, SumSquaredError, MeanSquaredError, \
    DifferentError, AbsoluteError
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier
from report.evaluator import Evaluator

from sklearn.metrics import accuracy_score

import sys

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='bce', learningRate=0.01, weightDecay=0.00005, epochs=50):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learningRate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.outputActivation = outputActivation
        self.weightDecay = weightDecay
        # self.cost = cost

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        
        if loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        elif loss == 'ce':
            self.loss = CrossEntropyError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.training_performances = []
        self.validation_performances = []

        self.layers = layers

        # Build up the network from specific layers
        if layers is None:
            self.layers = []

            # Input layer
            inputActivation = "sigmoid"
            self.layers.append(LogisticLayer(train.input.shape[1], 128,
                               None, inputActivation, False))

            # Output layer
            outputActivation = "softmax"
            self.layers.append(LogisticLayer(128, 10,
                               None, outputActivation, True))
        else:
            self.layers = []

            # Input layer
            inputActivation = "sigmoid"
            self.layers.append(LogisticLayer(train.input.shape[1], layers,
                                             None, inputActivation, False))

            # Output layer
            outputActivation = "softmax"
            self.layers.append(LogisticLayer(layers, 10,
                                             None, outputActivation, True))

        self.inputWeights = inputWeights

        # The code below was moved outside of the mlp in order to avoid reloading the data set every time a new MLP is tested
        """
        # add bias values ("1"s) at the beginning of all data sets
        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
                                            axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
                                              axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)
        """

    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        output = inp
        for layer in self.layers:
            output = np.insert(output, 0, 1, axis=0)
            output = layer.forward(output)
        return output
    
    def _update_weights(self, learningRate, error):
        """
        Update the weights of the layers by propagating back the error
        """
        derivatives = []
        weights = []

        # calculate derivatives
        for i, layer in enumerate(reversed(self.layers)):
            if i == 0:
                layer.computeDerivative(error, 1.0)
            else:
                layer.computeDerivative(derivatives[-1], weights[-1][1:])
            derivatives.append(layer.deltas)
            weights.append(layer.weights)

        # update weights
        for layer in reversed(self.layers):
            layer.updateWeights(learningRate, self.weightDecay)

    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        for epoch in range(self.epochs):
            for inp, label in zip(self.trainingSet.input, self.trainingSet.label):
                onehot_label = self.digit_to_onehot(label)
                # execute forward pass
                prediction = self._feed_forward(inp)
                # perform error computation
                prediction_error = self.loss.calculateDerivative(onehot_label, prediction)
                # apply weight updates
                self._update_weights(self.learningRate, prediction_error)

            train_accuracy = accuracy_score(self.trainingSet.label, self.evaluate(self.trainingSet.input))
            validation_accuracy = accuracy_score(self.validationSet.label, self.evaluate(self.validationSet.input))

            self.validation_performances.append(validation_accuracy)
            self.training_performances.append(train_accuracy)

            if verbose:
                print("Epoch %i of %i" % (epoch+1, self.epochs))
                print("Training accuracy: {0:.2f}%".format(train_accuracy*100))
                print("Validation accuracy: {0:.2f}%".format(validation_accuracy*100))
        pass

    def classify(self, test_instance):
        digit = np.argmax(self._feed_forward(test_instance))
        return digit

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)

    def digit_to_onehot(self, digit):
        return np.eye(10)[digit]
