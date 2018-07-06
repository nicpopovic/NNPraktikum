import matplotlib.pyplot as plt


class PerformancePlot(object):
    '''
    Class to plot the performances
    Very simple and badly-styled
    '''

    def __init__(self, name):
        '''
        Constructor
        '''
        self.name = name

    def draw_performance_epoch(self, training_performances, validation_performances, epochs, savename=None):
        plt.clf()
        plt.plot(range(epochs), training_performances, 'k', label="training")
        plt.plot(range(epochs), validation_performances, 'b', label="validation")
        plt.title("Performance of " + self.name + " over the epochs")
        plt.ylim(ymax=1)
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        if savename:
            plt.savefig(savename)
        else:
            plt.show()
