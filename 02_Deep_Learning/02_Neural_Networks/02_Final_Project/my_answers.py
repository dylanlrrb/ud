import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        self.activation_function = sigmoid

        self.activation_derivative = lambda x: sigmoid(x) * (1.0 - sigmoid(x))
        

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)

            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)

        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        ### Forward pass ###
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_outputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        ### Backward pass ###
        output_error = final_outputs - y
        output_error = np.expand_dims(output_error, axis=1)

        hidden_outputs = np.expand_dims(hidden_outputs, axis=1).T
        delta_h_o = output_error.dot(hidden_outputs)

        delta_weights_h_o += delta_h_o.T
        
        X = np.expand_dims(X, axis=1)
        hidden_error = self.weights_hidden_to_output.dot(output_error)
        z = X.T.dot(self.weights_input_to_hidden)
        delta =  hidden_error.T * self.activation_derivative(z)
        delta_i_h = delta.T.dot(X.T)

        delta_weights_i_h += delta_i_h.T

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        
        # TODO: Update the weights with gradient descent step
        self.weights_hidden_to_output -= (self.lr / n_records) * delta_weights_h_o
        self.weights_input_to_hidden -= (self.lr / n_records) * delta_weights_i_h

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''

        hidden_inputs = features.dot(self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_outputs = hidden_outputs.dot(self.weights_hidden_to_output) 
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 5000
learning_rate = 0.5
hidden_nodes = 30
output_nodes = 1
