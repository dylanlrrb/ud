### MNIST Data

The MNIST database is arguably the most famous database in the field of deep learning! Check out [this figure](https://www.kaggle.com/benhamner/popular-datasets-over-time) that shows datasets referenced over time in [NIPS](https://nips.cc/) papers.



## Normalizing image inputs

Data normalization is an important pre-processing step. It ensures that each input (each pixel value, in this case) comes from a standard distribution. That is, the range of pixel values in one input image are the same as the range in another image. This standardization makes our model train and reach a minimum error, faster!

Data normalization is typically done by subtracting the mean (the average of all pixel values) from each pixel, and then dividing the result by the standard deviation of all the pixel values. Sometimes you'll see an approximation here, where we use a mean and standard deviation of 0.5 to center the pixel values. [Read more about the Normalize transformation in PyTorch.](https://pytorch.org/docs/0.4.0/torchvision/transforms.html#transforms-on-torch-tensor)

The distribution of such data should resemble a [Gaussian function](http://mathworld.wolfram.com/GaussianFunction.html) centered at zero. For image inputs we need the pixel numbers to be positive, so we often choose to scale the data in a normalized range \[0,1\].



### ReLU Activation Function

The purpose of an activation function is to scale the outputs of a layer so that they are a consistent, small value. Much like normalizing input values, this step ensures that our model trains efficiently!

A ReLU activation function stands for "Rectified Linear Unit" and is one of the most commonly used activation functions for hidden layers. It is an activation function, simply defined as the **positive** part of the input, `x`. So, for an input image with any _negative_ pixel values, this would turn all those values to `0`, black. You may hear this referred to as "clipping" the values to zero; meaning that is the lower bound.

![](https://video.udacity-data.com/topher/2018/September/5ba9537e_relu-ex/relu-ex.png)

## Cross-Entropy Loss

In the [PyTorch documentation](https://pytorch.org/docs/stable/nn.html#crossentropyloss), you can see that the cross entropy loss function actually involves two steps:

- It first applies a softmax function to any output is sees
- Then applies [NLLLoss](https://pytorch.org/docs/stable/nn.html#nllloss); negative log likelihood loss

Then it returns the average loss over a batch of data. Since it applies a softmax function, we _do not_ have to specify that in the `forward` function of our model definition, but we could do this another way.

#### Another approach

We could separate the softmax and NLLLoss steps.

- In the `forward` function of our model, we would _explicitly_ apply a softmax activation function to the output, `x`.

```
 ...
 ...
# a softmax layer to convert 10 outputs into a distribution of class probabilities
x = F.log_softmax(x, dim=1)

return x
```

- Then, when defining our loss criterion, we would apply NLLLoss

```
# cross entropy loss combines softmax and nn.NLLLoss() in one single class
# here, we've separated them
criterion = nn.NLLLoss()
```

This separates the usual `criterion = nn.CrossEntropy()` into two steps: softmax and NLLLoss, and is a useful approach should you want the output of a model to be class _probabilities_ rather than class scores.

### Edge Handling

Kernel convolution relies on centering a pixel and looking at it's surrounding neighbors. So, what do you do if there are no surrounding pixels like on an image corner or edge? Well, there are a number of ways to process the edges, which are listed below. It’s most common to use padding, cropping, or extension. In extension, the border pixels of an image are copied and extended far enough to result in a filtered image of the same size as the original image.

**Extend** The nearest border pixels are conceptually extended as far as necessary to provide values for the convolution. Corner pixels are extended in 90° wedges. Other edge pixels are extended in lines.

**Padding** The image is padded with a border of 0's, black pixels.

**Crop** Any pixel in the output image which would require values from beyond the edge is skipped. This method can result in the output image being slightly smaller, with the edges having been cropped.

### Optional Resource

- Check out [this website](http://setosa.io/ev/image-kernels/), which allows you to create your own filter. You can then use your webcam as input to a convolutional layer and visualize the corresponding activation map! (#idea)

## Other kinds of pooling

Alexis mentioned one other type of pooling, and it is worth noting that some architectures choose to use [average pooling](https://pytorch.org/docs/stable/nn.html#avgpool2d), which chooses to average pixel values in a given window size. So in a 2x2 window, this operation will see 4 pixel values, and return a single, average of those four values, as output!

This kind of pooling is typically not used for image classification problems because maxpooling is better at noticing the most important details about edges and other features in an image, but you may see this used in applications for which _smoothing_ an image is preferable.

### Padding

Padding is just adding a border of pixels around an image. In PyTorch, you specify the size of this border.

Why do we need padding?

When we create a convolutional layer, we move a square filter around an image, using a center-pixel as an anchor. So, this kernel cannot perfectly overlay the edges/corners of images. The nice feature of padding is that it will allow us to control the spatial size of the output volumes (most commonly as we’ll see soon we will use it to exactly preserve the spatial size of the input volume so the input and output width and height are the same).

The most common methods of padding are padding an image with all 0-pixels (zero padding) or padding them with the nearest pixel value. You can read more about calculating the amount of padding, given a kernel\_size, [here](http://cs231n.github.io/convolutional-networks/#conv).

### PyTorch Layer Documentation

#### Convolutional Layers

We typically define a convolutional layer in PyTorch using `nn.Conv2d`, with the following parameters, specified:

```
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
```

- `in_channels` refers to the depth of an input. For a grayscale image, this depth = 1
- `out_channels` refers to the desired depth of the output, or the number of filtered images you want to get as output
- `kernel_size` is the size of your convolutional kernel (most commonly 3 for a 3x3 kernel)
- `stride` and `padding` have default values, but should be set depending on how large you want your output to be in the spatial dimensions x, y

[Read more about Conv2d in the documentation](https://pytorch.org/docs/stable/nn.html#conv2d).

#### Pooling Layers

Maxpooling layers commonly come after convolutional layers to shrink the x-y dimensions of an input, read more about pooling layers in PyTorch, [here](https://pytorch.org/docs/stable/nn.html#maxpool2d).


### Optional Resources

- Check out the [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) paper!
- Read more about [VGGNet](https://arxiv.org/pdf/1409.1556.pdf) here.
- The [ResNet](https://arxiv.org/pdf/1512.03385v1.pdf) paper can be found here.
- Here's the Keras [documentation](https://keras.io/applications/) for accessing some famous CNN architectures.
- Read this [detailed treatment](http://neuralnetworksanddeeplearning.com/chap5.html) of the vanishing gradients problem.
- Here's a GitHub [repository](https://github.com/jcjohnson/cnn-benchmarks) containing benchmarks for different CNN architectures.
- Visit the [ImageNet Large Scale Visual Recognition Competition (ILSVRC)](http://www.image-net.org/challenges/LSVRC/) website.

### (REALLY COOL) Optional Resources
(#idea a bunch of them)
If you would like to know more about interpreting CNNs and convolutional layers in particular, you are encouraged to check out these resources:

- Here's a [section](http://cs231n.github.io/understanding-cnn/) from the Stanford's CS231n course on visualizing what CNNs learn.
- Check out this [demonstration](https://aiexperiments.withgoogle.com/what-neural-nets-see) of a cool [OpenFrameworks](http://openframeworks.cc/) app that visualizes CNNs in real-time, from user-supplied video!
- Here's a [demonstration](https://www.youtube.com/watch?v=AgkfIQ4IGaM&t=78s) of another visualization tool for CNNs. If you'd like to learn more about how these visualizations are made, check out this [video](https://www.youtube.com/watch?v=ghEmQSxT6tw&t=5s).
- Read this [Keras blog post](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html) on visualizing how CNNs see the world. In this post, you can find an accessible introduction to Deep Dreams, along with code for writing your own deep dreams in Keras. When you've read that:
    
    - Also check out this [music video](https://www.youtube.com/watch?v=XatXy6ZhKZw) that makes use of Deep Dreams (look at 3:15-3:40)!
    - Create your own Deep Dreams (without writing any code!) using this [website](https://deepdreamgenerator.com/).
- If you'd like to read more about interpretability of CNNs:
    
    - Here's an [article](https://blog.openai.com/adversarial-example-research/) that details some dangers from using deep learning models (that are not yet interpretable) in real-world applications.
    - There's a lot of active research in this area. [These authors](https://arxiv.org/abs/1611.03530) recently made a step in the right direction.

## Other Links

[https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/](https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/)

[https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215)

[https://cs231n.github.io/convolutional-networks/](https://cs231n.github.io/convolutional-networks/)

[https://www.youtube.com/c/GeneKogan/videos](https://www.youtube.com/c/GeneKogan/videos)

[https://github.com/vdumoulin/conv_arithmetic](https://github.com/vdumoulin/conv_arithmetic)