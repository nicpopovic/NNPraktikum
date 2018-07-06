#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot
import numpy as np


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                      oneHot=False)

    # This was moved outside of the mlp in order to avoid reloading the data set every time a new MLP is tested
    # add bias values ("1"s) at the beginning of all data sets
    data.trainingSet.input = np.insert(data.trainingSet.input, 0, 1, axis=1)
    data.validationSet.input = np.insert(data.validationSet.input, 0, 1, axis=1)
    data.testSet.input = np.insert(data.testSet.input, 0, 1, axis=1)

    myMLPClassifier = MultilayerPerceptron(data.trainingSet,
                                           data.validationSet,
                                           data.testSet,
                                           learningRate=0.005,
                                           epochs=2,
                                           weightDecay=0.00005,
                                           loss="ce")

    # Report the result #
    print("=========================")
    evaluator = Evaluator()

    # Train the classifiers
    print("=========================")
    print("Training..")

    print("\nMLP has been training..")
    myMLPClassifier.train(verbose=True)
    print("Done..")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    mlpPred = myMLPClassifier.evaluate()

    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("\nResult of the MLP recognizer:")
    # evaluator.printComparison(data.testSet, lrPred)
    evaluator.printAccuracy(data.testSet, mlpPred)
    evaluator.plotAccuracyHistogramm(data.testSet, mlpPred)

    # Draw
    plot = PerformancePlot("MLP validation")
    plot.draw_performance_epoch(myMLPClassifier.training_performances, myMLPClassifier.validation_performances, myMLPClassifier.epochs)


if __name__ == '__main__':
    main()
