#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.mlp import MultilayerPerceptron

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot
import numpy as np
import GPyOpt


def run_MLP_with_hyperparameters(hypers):
    hiddenLayerSize, learningRate, decay = hypers[0]
    global step
    step += 1
    # initalize MLP with hyperparameters
    myMLPClassifier = MultilayerPerceptron(data.trainingSet,
                                           data.validationSet,
                                           data.testSet,
                                           learningRate=learningRate,
                                           epochs=50,
                                           weightDecay=decay,
                                           layers=int(hiddenLayerSize),
                                           loss="ce")
    print("=========================")
    print("Training with layers %i, learning rate: %.2f, weightDecay:%.2f" % (int(hiddenLayerSize), learningRate, decay))
    # train it
    myMLPClassifier.train(verbose=True)
    print("Done..")

    # evaluate configuration
    mlpPred = myMLPClassifier.evaluate()
    evaluator = Evaluator()
    test_accuracy = evaluator.returnAccuracy(data.testSet, mlpPred)
    training_accuracy = myMLPClassifier.training_performances[-1]
    validation_accuracy = myMLPClassifier.validation_performances[-1]

    # write to log
    text_file = open(path_textfile, "a")
    text_file.write("|" + str(int(step)) + "|" + str(int(hiddenLayerSize)) + "|" + str(learningRate) + "|" + str(decay) + "|" +
                    str("%.2f %%" % np.multiply(training_accuracy, 100)) + "|" +
                    str("%.2f %%" % np.multiply(validation_accuracy, 100)) + "|" +
                    str("%.2f %%" % np.multiply(test_accuracy, 100)) + "|" + "\n")
    text_file.close()

    # Draw and save plots
    evaluator.plotAccuracyHistogramm(data.testSet, mlpPred, savename=str("plots/histogram_%i.png"%step))
    plot = PerformancePlot("MLP validation")
    plot.draw_performance_epoch(myMLPClassifier.training_performances, myMLPClassifier.validation_performances, myMLPClassifier.epochs, savename=str("plots/plot_%i.png"%step))

    return myMLPClassifier.validation_performances[-1]


data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                  oneHot=False)

# This was moved outside of the mlp in order to avoid reloading the data set every time a new MLP is tested
# add bias values ("1"s) at the beginning of all data sets
data.trainingSet.input = np.insert(data.trainingSet.input, 0, 1, axis=1)
data.validationSet.input = np.insert(data.validationSet.input, 0, 1, axis=1)
data.testSet.input = np.insert(data.testSet.input, 0, 1, axis=1)

path_textfile = "log.md"
step = 0

# write to log
text_file = open(path_textfile, "a")
text_file.write('|step|hidden layer size|learning rate|weight decay|training accuracy|validation accuracy|test accuracy|'  + "\n")
text_file.write('|---|---|---|---|---|---|---|'  + "\n")
text_file.close()

# hyperparameter ranges
hyperparameter_ranges = [{'name': 'hiddenLayerSize', 'type': 'discrete', 'domain': [8, 16, 32, 64, 128, 256]},
                         {'name': 'learningRate', 'type': 'continuous', 'domain': (1e-8, 0.1)},
                         {'name': 'decay', 'type': 'continuous', 'domain': (1e-6, 1e-4)}]

# initialize optimizer
myBopt = GPyOpt.methods.BayesianOptimization(
    f=run_MLP_with_hyperparameters,             # Objective function
    domain=hyperparameter_ranges,               # Box-constrains of the problem
    initial_design_numdata=15,                  # Number of initial samples
    initial_design_type='random',               # Defines how initial samples are chosen
    acquisition_type='EI',                      # Expected Improvement
    exact_feval=False                           # Assume sample is noisy
    )

max_iter = 30       ## maximum number of iterations

# run optimizer
myBopt.run_optimization(max_iter, eps=0)
