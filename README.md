# Numpy_FC_NN

## The sequential layer receives a list of layers for the constructor argument, and calls them in order on forward
## and in reverse order on backward. It is just syntactic sugar that makes the code much nicer. For example, you can
## create a two-layer neural network with tanh activation functions with net = Sequential([Linear(5, 8), Tanh(),
## Linear(8, 3), Tanh()]) and then run a forward pass using output = net.forward(your data).
##
## All the modules have a forward() and a backward() function. forward() receives one argument (except for the loss) and
## returns that layer's output. The backward() receives the dL/dout, flowing back on the output of the layer, and
## should return a BackwardResult object with the following fields:
##      variable_grads: a dict of gradients, where the keys are the same as in the keys in the layer's .var. The
##                      values are numpy arrays representing the gradient for that variable.
##      input_grads: a numpy array representing the gradient for the input (that was passed to the layer in the forward
##                   pass).
##
## The backward() does not receive the forward pass's input, although it might be needed for the gradient
## calculation. You should save them in the forward pass for later use in the backward pass. You don't have to worry
## about most of this, as it is already implemented in the skeleton. There are 2 important takeaways: you have to
## calculate the gradient of both of your variables and the layer input in the backward pass, and if you need to reuse
## the input from the forward pass, you need to save it.
##
