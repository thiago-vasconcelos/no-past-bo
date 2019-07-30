import GPyOpt
import GPy
import GPyOpt.models
import GPyOpt.core
from GPyOpt.acquisitions import AcquisitionEI, AcquisitionMPI, AcquisitionLCB
from GPyOpt.experiment_design import initial_design

class BO(GPyOpt.methods.ModularBayesianOptimization):
    def suggest_next_locations(self, context = None, pending_X = None, ignored_X = None):
        self.model_parameters_iterations = None
        self.num_acquisitions = 0
        self.context = context
    #     self._update_model(self.normalization_type)

        suggested_locations = self._compute_next_evaluations(pending_zipped_X = pending_X, ignored_zipped_X = ignored_X)

        return suggested_locations

def normalize(value):
    return (value - value.mean())/value.std()


def getProbs(scores):
    dmax = scores.max()
    dmin = scores.min()
    if(dmax == dmin):
        aux = np.exp(np.zeros(3))
    else:
        aux = np.exp(eta * (scores - dmax)/(dmax-dmin))
    return  aux/aux.sum()

def chooseHedge(scores):
    probs = getProbs(scores)
    cumsum = probs.cumsum()
    aux = np.random.uniform(0, cumsum[-1])
    return np.argmax(cumsum > aux)

def get_best_evaluation(X_init, y_init, bo, factor=1., iterations=10):
    for j in range(iterations):
        

