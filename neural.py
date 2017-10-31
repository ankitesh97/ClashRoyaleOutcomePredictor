
from layer import Layer, NDI
#this is neural.py, contains class for neural network of the input encoder, as well as for the match predictor

# this uses the class Layer in layer.py
class NNTrad:
    '''
    input_dim : is the input dimension that is to be fed
    output_dim: is the output dimension (that you want the input encoded in)
    n_hidden: number of hidden layers
    hidden_sizes: list of hidden nodes sizes 
    note: in the last layer no nonlinearity will be applied
    '''
    def __init__ (self, input_dim, output_dim, n_hidden, hidden_sizes, nonlin, nonlin_deriv, last_nonlin=None, last_nonlin_deriv=None, identity=None, last_linear_flag=False):
        self.layers = []
        dimensions = [input_dim] + hidden_sizes + [output_dim]
        for i in range(n_hidden):
            l = Layer(dimensions[i], dimensions[i+1], nonlin, nonlin_deriv)
            self.layers.append(l)
        
        #add the last layer
        if last_linear_flag == True:
            self.layers.append(Layer(dimensions[-2], dimensions[-1], identity, identity))
        else:
            if last_nonlin:
                self.layers.append(Layer(dimensions[-2], dimensions[-1], last_nonlin, last_nonlin_deriv))
            else:
                self.layers.append(Layer(dimensions[-2], dimensions[-1], nonlin, nonlin_deriv))
        

    
    
    #forward
    def forward(self,input):

        inp = input
        for l in self.layers:
            inp = l.forward(inp)
            
        return self.layers[-1].output
    
    #backward
    def backprop(self,d_out):
        dell_out = d_out
        for l in self.layers[::-1]:
            dell_out = l.backward(dell_out)
            l.update()
            

            
# this class of neural network is for the main which predicts the output this will use the ndi interface

class NNNdi:
    def __init__(self, input_dim, output_dim, n_hidden, hidden_sizes, nonlin, nonlin_deriv, last_nonlin=None, last_nonlin_deriv=None):
        self.layers = []
        dimensions = [input_dim] + hidden_sizes + [output_dim]
        for i in range(n_hidden):
            l = NDI(dimensions[i], dimensions[i+1], nonlin, nonlin_deriv)
            self.layers.append(l)
        
        #add the last layer

        if last_nonlin:
            self.layers.append(Layer(dimensions[-2], dimensions[-1], last_nonlin, last_nonlin_deriv))
        else:
            self.layers.append(Layer(dimensions[-2], dimensions[-1], nonlin, nonlin_deriv))
        self.cache = []

    
    def forward(self,input):
        inp = input
        for l in self.layers:
            inp, grads = l.forward(inp)
            self.cache.append(grads)
        return self.layers[-1].output
    
    
    def backward_synthetic_weight_updates(self,d_out):
        true_grad = self.layers[-1].normal_update(d_out)
        for l in range(-2,-1,-1):
            self.layers[l].update_synthetic_weights(true_grad)
            true_grad = self.cache[l]
            
    
        
            
