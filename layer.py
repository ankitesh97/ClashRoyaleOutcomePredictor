#layer.py

alpha = 0.1
alpha_synthetic = 0.1
# old fashioned layer of nn
class Layer:
    def __init__(self,input_dim, output_dim, nonlin, nonlin_deriv):
        self.weights = np.random.randn(output_dim,input_dim) * np.sqrt(1.0/(1+input_dim))
        self.bias = np.random.randn(output_dim) * np.sqrt(1.0/1+input_dim)
        self.nonlin = nonlin
        self.nonlin_deriv = nonlin_deriv
        
    
    def forward(self, input):
        self.input = input
        self.output = self.nonlin(np.dot(self.weights, input) + self.bias)
        return self.output
    
    def backward(self,d_out):
        self.output_delta = d_out * self.nonlin_deriv(self.output)
        return np.dot(self.output_delta.reshape(1,-1), self.weights)
        
    def update(self):
        self.weights -= np.outer(self.output_delta, self.input) * alpha
        self.bias -= self.output_delta * alpha
        

class NDI:
    def __init__(self, input_dim, output_dim, nonlin, nonlin_deriv):
        self.weights = np.random.randn(output_dim,input_dim) * np.sqrt(1.0/(1+input_dim))
        self.bias = np.random.randn(output_dim) * np.sqrt(1.0/1+input_dim)
        self.nonlin = nonlin
        self.nonlin_deriv = nonlin_deriv
        
        # weights of synthetic gradient networks
        self.weights_synthetic = np.random.randn(output_dim,output_dim) * np.sqrt(1.0/1 + output_dim)
    
    def forward(self, input):
        self.input = input
        self.output = self.nonlin(np.dot(self.weights, input) + self.bias)
        
        #calculate synthetic error
        self.synthetic_gradient = np.dot(self.weights_synthetic, self.output)
        
        self.output_delta_synthetic = self.synthetic_gradient * self.nonlin_deriv(self.output)
        #update
        self.weights -= np.outer(self.output_delta_synthetic, self.input) * alpha
        self.bias -= self.output_delta * alpha
        return self.output, np.dot(self.output_delta_synthetic.reshape(1,-1), self.weights)
    
    def update_synthetic_weights(self, true_gradient):
        change = self.synthetic_gradient - self.true_gradient
        self.weights_synthetic -= np.outer(change, self.output) * alpha_synthetic
        
        
    def normal_update(self,d_out):
        self.weights += np.outer(self.output_delta_synthetic, self.input) * alpha
        self.bias += self.output_delta * alpha
        output_delta = d_out * self.nonlin_deriv(self.output)
        to_send = np.dot(self.output_delta.reshape(1,-1), self.weights)
        self.weights -= np.outer(output_delta, self.input) * alpha
        self.bias -= output_delta * alpha
        return to_send
