import GPyOpt
import GPy
import GPyOpt.models
import GPyOpt.core
import numpy as np
from GPyOpt.acquisitions import AcquisitionEI, AcquisitionMPI, AcquisitionLCB

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


def getProbs(scores, eta):
    dmax = scores.max()
    dmin = scores.min()
    if(dmax == dmin):
        aux = np.exp(np.zeros(3))
    else:
        aux = np.exp(eta * (scores - dmax)/(dmax-dmin))
    return  aux/aux.sum()

def chooseHedge(scores, eta):
    probs = getProbs(scores, eta)
    cumsum = probs.cumsum()
    aux = np.random.uniform(0, cumsum[-1])
    return np.argmax(cumsum > aux)

def build_acquisition(X_init, space, aquisition_function, model):
    aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space, eps=0)
    if(aquisition_function['type'] == 'ei'):
        aquisition_function = AcquisitionEI(model=model, space=space, optimizer=aquisition_optimizer,jitter=aquisition_function['epsilon'])
    elif(aquisition_function['type']== 'pi'):
        aquisition_function = AcquisitionMPI(model=model, space=space, optimizer=aquisition_optimizer,jitter=aquisition_function['epsilon'])
    elif(aquisition_function['type'] == 'lcb'):
        lcb_const = np.sqrt( aquisition_function['upsilon']* (2*  np.log( ((X_init.shape[0])**(X_init.shape[1]/2. + 2))*(np.pi**2)/(3. * aquisition_function['delta'])  )))
        aquisition_function = AcquisitionLCB(model=model, space=space, optimizer=aquisition_optimizer,exploration_weight=lcb_const)
    return aquisition_function


def build_bos(X_init, y_init, model, space, aquisition_functions):    
    bos = []
    for function in aquisition_functions:
        aquisition_function = build_acquisition(X_init, space, function, model)
        evaluator = GPyOpt.core.evaluators.Sequential(aquisition_function)
        bo = BO(model, space, None, aquisition_function, evaluator, X_init=X_init, Y_init=y_init)
        bos.append(bo)
    return bos

def get_best_evaluation(X_init, y_init, space, acquisitions, optimization_function, factor=1., iterations=10, eta=4):
    assert(X_init.shape[0] == y_init.shape[0])
    initial_data_size = X_init.shape[0]
    portfolio_size = len(acquisitions)
    scores_list=[]
    scores = np.zeros(portfolio_size)
    x = [None] * portfolio_size
    previous_x = [None] * portfolio_size

    X = X_init
    y = y_init
    for i in range(iterations):
        kernel = GPy.kern.Matern52(input_dim=X_init.shape[1], ARD=True)
        model = GPyOpt.models.GPModel(kernel=kernel,optimize_restarts=5,verbose=False, ARD=True, exact_feval=True, max_iters=5000)
        normalized_y = normalize(y_init)
        model.updateModel(X_init,normalized_y,None,None)
        bos = build_bos(X, y, model, space, acquisitions)

        if(i!=0):
            for j in range(portfolio_size):
                previous_x[j] = x[j]

        for j in range(portfolio_size):
            x[j] = bos[j].suggest_next_locations()

        if(i!=0):
            for j in range(portfolio_size):
                scores[j] = factor*scores[j] - (model.predict(previous_x[j])[0]*y.std() + y.mean())
            scores_list.append(scores.copy())

        randomChoice = chooseHedge(scores, eta)
        bestChoice = x[randomChoice]


        x_next = bestChoice
        y_next = optimization_function(x_next)

        X = np.vstack((X, x_next))
        y = np.vstack((y, y_next))
            

    execution = np.concatenate((X,y), axis=1)
    for j in range(portfolio_size):
        scores[j] = factor*scores[j] - (model.predict(previous_x[j])[0]*y.std() + y.mean())
    scores_list.append(scores.copy())

    return execution, scores_list



