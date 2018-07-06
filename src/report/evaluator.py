# -*- coding: utf-8 -*-

from __future__ import division
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class Evaluator:
    """
    Print performance of a classification model over a dataset
    """

    def printTestLabel(self, testSet):
        # print all test labels
        for label in testSet.label:
            print(label)

    def printResultedLabel(self, pred):
        # print all test labels
        for result in pred:
            print(result)

    def printComparison(self, testSet, pred):
        for label, result in zip(testSet.label, pred):
            print("Label: %r. Prediction: %r" % (bool(label), bool(result)))

    def printClassificationResult(self, testSet, pred, targetNames):
        print(classification_report(testSet.label,
                                    pred,
                                    target_names=targetNames))

    def printConfusionMatrix(self, testSet, pred):
        print(confusion_matrix(testSet.label, pred))

    def printAccuracy(self, testSet, pred):
        print("Accuracy of the recognizer: %.2f%%" %
              (accuracy_score(testSet.label, pred)*100))

    def returnAccuracy(self, testSet, pred):
        return accuracy_score(testSet.label, pred)

    def plotAccuracyHistogramm(self, testSet, pred, savename=None):
        correct = [0 for i in range(10)]
        total = [0 for i in range(10)]

        for label, result in zip(testSet.label, pred):
            total[label] = total[label] + 1
            if label == result:
                correct[label] = correct[label] + 1

        histogram = [100*cr/tot for cr, tot in zip(correct, total)]

        fig, ax = plt.subplots()
        rects1 = ax.bar(range(10), histogram)
        ax.set_xlabel('Digit')
        ax.set_ylabel('Accuracy [%]')
        ax.set_title('Accuracy by digit')
        ax.set_xticks(range(10))
        ax.set_ylim(0, 100)
        if savename:
            plt.savefig(savename)
        else:
            plt.show()
