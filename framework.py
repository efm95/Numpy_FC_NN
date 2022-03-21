import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from numpy.core.fromnumeric import shape

from numpy.lib.arraysetops import unique

# Interface definitions
class Layer:
    var: Dict[str, np.ndarray] = {}

    @dataclass
    class BackwardResult:
        variable_grads: Dict[str, np.ndarray]
        input_grads: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, error: np.ndarray) -> BackwardResult:
        raise NotImplementedError()


class Loss:
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError()

    def backward(self) -> np.ndarray:
        raise NotImplementedError()


class Tanh(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        ## Implement

        result = np.tanh(x)
        #result = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        ## End
        self.saved_variables = {
            "result": result
        }
        return result

    def backward(self, grad_in: np.ndarray) -> Layer.BackwardResult:
        tanh_x = self.saved_variables["result"]


        d_x = (1-tanh_x**2)*grad_in
        
        #Alternativelty
        #d_x = np.matmul(1 - ((np.exp(tanh_x) - np.exp(- tanh_x) ** 2)  / (np.exp(tanh_x) + np.exp(- tanh_x) ** 2)), grad_in)
        
        assert d_x.shape == tanh_x.shape, "Input: grad shape differs: %s %s" % (d_x.shape, tanh_x.shape)

        self.saved_variables = None
        return Layer.BackwardResult({}, d_x)


class Softmax(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        
        result = np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)
        
        #IN CASE OF NUMERICAL INSTABILITY
        #result = np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x))) 
        
        self.saved_variables = {
            "result": result
        }
        return result

    def backward(self, grad_in: np.ndarray) -> Layer.BackwardResult:
        softmax = self.saved_variables["result"]

        #d_x =  np.zeros(softmax.shape) # <- easier to interpret but ugly to see
        #for i in range(d_x.shape[0]):
        #    tmp_soft = softmax[i]
        #    tmp_d_x = np.diag(tmp_soft) - np.outer(tmp_soft, tmp_soft.T)
        #    d_x[i] = np.matmul(tmp_d_x, grad_in[i])
        
        d_x = softmax * (grad_in - (grad_in * softmax).sum(axis=1, keepdims=True)) # <- more compact but harder to interpret

        assert d_x.shape == softmax.shape, "Input: grad shape differs: %s %s" % (d_x.shape, softmax.shape)

        self.saved_variables = None
        return Layer.BackwardResult({}, d_x)


class Linear(Layer):
    def __init__(self, input_size: int, output_size: int):
        self.var = {
            "W": np.random.normal(0, np.sqrt(2 / (input_size + output_size)), (input_size, output_size)),
            "b": np.zeros((output_size), dtype=np.float32)
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        W = self.var['W']
        b = self.var['b']

        y = np.matmul(x,W)+b
        self.saved_variables = {
            "input": x,
            "W":W,
            "b":b
        }

        return y

    def backward(self, grad_in: np.ndarray) -> Layer.BackwardResult:

        x = self.saved_variables["input"]
        W = self.saved_variables["W"]
        b = self.saved_variables["b"]

        dW = np.matmul((x).T,grad_in)
        db = np.matmul(np.ones(x.shape[0]),grad_in)
        
        d_inputs = np.matmul(grad_in,W.T)

        assert d_inputs.shape == x.shape, "Input: grad shape differs: %s %s" % (d_inputs.shape, x.shape)
        assert dW.shape == self.var["W"].shape, "W: grad shape differs: %s %s" % (dW.shape, self.var["W"].shape)
        assert db.shape == self.var["b"].shape, "b: grad shape differs: %s %s" % (db.shape, self.var["b"].shape)

        self.saved_variables = None
        updates = {"W": dW,
                   "b": db}
        return Layer.BackwardResult(updates, d_inputs)


class Sequential(Layer):
    class RefDict(dict):
        def __setitem__(self, k, v):
            assert k in self, "Trying to set a non-existing variable %s" % k
            ref = super().__getitem__(k)
            ref[0][ref[1]] = v

        def __getitem__(self, k):
            ref = super().__getitem__(k)
            return ref[0][ref[1]]

        def items(self) -> Tuple[str, np.ndarray]:
            for k in self.keys():
                yield k, self[k]

    def __init__(self, list_of_modules: List[Layer]):
        self.modules = list_of_modules

        refs = {}
        for i, m in enumerate(self.modules):
            refs.update({"mod_%d.%s" % (i,k): (m.var, k) for k in m.var.keys()})

        self.var = self.RefDict(refs)

    def forward(self, input: np.ndarray) -> np.ndarray:
        
        x = input 
        for i in self.modules:
            x=i.forward(x)
            
        return x

    def backward(self, grad_in: np.ndarray) -> Layer.BackwardResult:
        variable_grads = {}

        for module_index in reversed(range(len(self.modules))):
            module = self.modules[module_index]
            
            grads = module.backward(grad_in)

            ## End
            grad_in = grads.input_grads
            variable_grads.update({"mod_%d.%s" % (module_index, k): v for k, v in grads.variable_grads.items()})

        return Layer.BackwardResult(variable_grads, grad_in)


class CrossEntropy(Loss):
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        Y = prediction
        T = target
        n = prediction.size
        
        if len(T.shape)>1:
            ce = (np.where(T==1, -np.log(Y), 0)).sum(axis=1)
            mean_ce = np.mean(ce)
        else:
            mean_ce = np.mean(-np.log(Y[range(Y.shape[0]), T]))
            
        self.saved_variables={
            "Y":Y,
            "T":T,
            "n":n
        }

        return mean_ce

    def backward(self) -> np.ndarray:
        y = self.saved_variables['Y']
        t = self.saved_variables['T']
        n = self.saved_variables['n']

        if len(t.shape)>1:
            d_prediction = np.where(t==1, -1/y, 0)/y.shape[0]
        else:
            d_prediction = np.zeros(y.shape)
            d_prediction[range(y.shape[0]),t] = y[range(y.shape[0]),t]-1
            d_prediction = d_prediction/y.shape[0]
                           
        assert d_prediction.shape == y.shape, "Error shape doesn't match prediction: %d %d" % \
                                              (d_prediction.shape, y.shape)

        self.saved_variables = None
        return d_prediction


def train_one_step(model: Layer, loss: Loss, learning_rate: float, input: np.ndarray, target: np.ndarray) -> float:
    
    loss_value = loss.forward(model.forward(input),target)
    variable_gradients = model.backward(loss.backward()).variable_grads

    for key, item in variable_gradients.items():
        model.var[key] = model.var[key]-learning_rate*item

    return loss_value


def create_network() -> Layer:

    network = Sequential([Linear(2,50), Tanh(),Linear(50,30), Tanh(), Linear(30,2),Softmax()])

    return network


def gradient_check():
    X, T = twospirals(n_points=10)
    NN = create_network()
    eps = 0.0001

    loss = CrossEntropy()
    loss.forward(NN.forward(X), T)
    variable_gradients = NN.backward(loss.backward()).variable_grads

    all_succeeded = True

    for key, value in NN.var.items():
        variable = NN.var[key].reshape(-1)
        variable_gradient = variable_gradients[key].reshape(-1)
        success = True

        if NN.var[key].shape != variable_gradients[key].shape:
            print("[FAIL]: %s: Shape differs: %s %s" % (key, NN.var[key].shape, variable_gradients[key].shape))
            success = False
            break

        for index in range(variable.shape[0]):
            var_backup = variable[index]

            analytic_grad = variable_gradient[index]

            variable[index] = var_backup + eps
            ce_plus = loss.forward(NN.forward(X), T)

            variable[index] = var_backup - eps
            ce_minus = loss.forward(NN.forward(X), T)

            numeric_grad = (ce_plus - ce_minus) / (2*eps)

            variable[index] = var_backup
            if abs(numeric_grad - analytic_grad) > 0.00001:
                print("[FAIL]: %s: Grad differs: numerical: %f, analytical %f" % (key, numeric_grad, analytic_grad))
                success = False
                break

        if success:
            print("[OK]: %s" % key)

        all_succeeded = all_succeeded and success

    return all_succeeded


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    np.random.seed(12345)

    plt.ion()


    def twospirals(n_points=120, noise=1.6, twist=420):
        """
         Returns a two spirals dataset.
        """
        np.random.seed(0)
        n = np.sqrt(np.random.rand(n_points, 1)) * twist * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
        d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
        X, T = (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
                np.hstack((np.zeros(n_points), np.ones(n_points))))
        T = np.reshape(T, (T.shape[0], 1))
        T = np.concatenate([T, 1-T], axis=1)
        return X, T


    fig, ax = plt.subplots()


    def plot_data(X, T):
        ax.scatter(X[:, 0], X[:, 1], s=40, c=T[:, 0], cmap=plt.cm.Spectral)


    def plot_boundary(model, X, targets, threshold=0.0):
        ax.clear()
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        X_grid = np.c_[xx.ravel(), yy.ravel()]
        y = model.forward(X_grid)[:, 0]
        ax.contourf(xx, yy, y.reshape(*xx.shape) < threshold, alpha=0.5)
        plot_data(X, targets)
        ax.set_ylim([y_min, y_max])
        ax.set_xlim([x_min, x_max])
        plt.show()
        plt.draw()
        plt.pause(0.001)


    def main():
        print("Checking the network")
        if not gradient_check():
            print("Failed. Not training, because your gradients are not good.")
            return
        print("Done. Training...")

        X, T = twospirals(n_points=200, noise=1.6, twist=600)
        NN = create_network()
        loss = CrossEntropy()

        learning_rate = 0.02

        for i in range(20000):
            curr_error = train_one_step(NN, loss, learning_rate, X, T)
            if i % 200 == 0:
                print("step: ", i, " cost: ", curr_error)
                plot_boundary(NN, X, T, 0.5)

        plot_boundary(NN, X, T, 0.5)
        print("Done. Close window to quit.")
        plt.ioff()
        plt.show()



    main()
