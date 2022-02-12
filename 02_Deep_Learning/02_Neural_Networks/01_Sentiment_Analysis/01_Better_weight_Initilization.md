### Better Weight Initialization Strategy

In the last video, Andrew Trask built a neural network and initialized the weights according to the function `init_network`:

```
def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights

        # These are the weights between the input layer and the hidden layer.
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))

        # These are the weights between the hidden layer and the output layer.
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))

        # The input layer, a two-dimensional matrix with shape 1 x input_nodes
        self.layer_0 = np.zeros((1,input_nodes))
```

Here, I'd like you to take note of the line `self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, (self.hidden_nodes, self.output_nodes))` this is one possible initialization strategy, but there is actually a better way to initialize these weights that I will briefly go over in this concept.

### Different Initialization, Better Accuracy

You'll learn more about weight initialization, later on in this program. Suffice to say that the general rule for setting the weights in a neural network is to set them to be close to zero without being too small.

> Good practice is to start your weights in the range of **\[-y, y\]** where y\=1/n y=1/\\sqrt{n}y\=1/n ​ And **n** is the number of inputs to a given layer.

In _this_ case, between layers 1 and 2, a better solution would be to define the weights as a function of the number of nodes **n** in the _hidden_ layer.

So, the initialization code would change; you'd use `hidden_nodes**-0.5` rather than `output_nodes**-0.5` If you try this out, you should see that you get improved training with a larger `learning rate = 0.01` rather than waiting for a learning rate of 0.001.

```
def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights

        # These are the weights between the input layer and the hidden layer.
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))

        # These are the weights between the hidden layer and the output layer.
        ## NOTE: the difference in the standard deviation of the normal weights
        ## This was changed from `self.output_nodes**-0.5` to `self.hidden_nodes**-0.5`
        self.weights_1_2 = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))

        # The input layer, a two-dimensional matrix with shape 1 x input_nodes
        self.layer_0 = np.zeros((1,input_nodes))
```

![](https://video.udacity-data.com/topher/2018/November/5beb458b_screen-shot-2018-11-13-at-1.43.17-pm/screen-shot-2018-11-13-at-1.43.17-pm.png)

Results for learning rate = 0.01.

### The Code

Andrew will still use the initialization strategy using `output_nodes**-0.5`, but you can see the solution code for this _alternate_ weight initialization strategy in [the Github repo](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/sentiment-analysis-network/Sentiment_Classification_Solutions_2_Better_Weight_Initialization.ipynb).