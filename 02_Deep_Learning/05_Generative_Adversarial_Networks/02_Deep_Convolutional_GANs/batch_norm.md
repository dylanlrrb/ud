# What is Batch Normalization?

Batch normalization was introduced in Sergey Ioffe's and Christian Szegedy's 2015 paper [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf). The idea is that, instead of just normalizing the inputs to the network, we normalize the inputs to every layer _within_ the network.

### Batch Normalization

It's called "batch" normalization because, during **training**, we normalize each layer's inputs by using the _mean_ and _standard deviation_ (or variance) of the values in the current batch. These are sometimes called the **batch statistics**.

> Specifically, batch normalization normalizes the output of a previous layer by **subtracting the batch mean and dividing by the batch standard deviation**.

Why might this help? Well, we know that normalizing the inputs to a network helps the network learn and converge to a solution. However, a network is a series of layers, where the output of one layer becomes the input to another. That means we can think of any layer in a neural network as the _first_ layer of a smaller network.

### Normalization at Every Layer

For example, imagine a 3 layer network.

![](https://video.udacity-data.com/topher/2018/November/5be3855d_3-layers/3-layers.png)

3 layer network

Instead of just thinking of it as a single network with inputs, layers, and outputs, think of the output of layer 1 as the input to a two layer network. This two layer network would consist of layers 2 and 3 in our original network.

![](https://video.udacity-data.com/topher/2018/November/5be38631_2-layers/2-layers.png)

2 layer network

Likewise, the output of layer 2 can be thought of as the input to a single layer network, consisting only of layer 3.

![](https://video.udacity-data.com/topher/2018/November/5be38649_one-layer/one-layer.png)

One layer network

When you think of it like this - as a series of neural networks feeding into each other - then it's easy to imagine how normalizing the inputs to each layer would help. It's just like normalizing the inputs to any other neural network, but you're doing it at **every layer (sub-network)**.

## Internal Covariate Shift

Beyond the intuitive reasons, there are good mathematical reasons to motivate batch normalization. It helps combat what the authors call **internal covariate shift**.

> In this case, internal covariate shift refers to the change in the distribution of the inputs to different layers. It turns out that training a network is most efficient when the distribution of inputs to each layer is similar!

And batch normalization is one method of standardizing the distribution of layer inputs. This discussion is best handled [in the paper](https://arxiv.org/pdf/1502.03167.pdf) and in [Deep Learning](http://www.deeplearningbook.org/), a book you can read online written by Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Specifically, check out the batch normalization section of [Chapter 8: Optimization for Training Deep Models](http://www.deeplearningbook.org/contents/optimization.html).

* * *

# The Math

Next, let's do a deep dive into the math behind batch normalization. This is not critical for you to know, but it may help your understanding of this whole process!

## Getting the mean and variance

In order to normalize the values, we first need to find the average value for the batch. If you look at the code, you can see that this is not the average value of the batch _inputs_, but the average value coming _out_ of any particular layer before we pass it through its non-linear activation function and then feed it as an input to the _next_ layer.

We represent the average as **mu\_B** μB \\mu\_B μB​ which is simply the sum of all of the values, **x\_i** divided by the number of values, **m**.

μB←1m∑i\=1mxi \\mu\_B \\leftarrow \\frac{1}{m}\\sum\_{i=1}^m x\_i μB​←m1​i\=1∑m​xi​

We then need to calculate the variance, or mean squared deviation, represented as σB2\\sigma\_{B}^{2}σB2​

If you aren't familiar with statistics, that simply means for each value **x\_i**, we subtract the average value (calculated earlier as **mu\_B**), which gives us what's called the "deviation" for that value. We square the result to get the squared deviation. Sum up the results of doing that for each of the values, then divide by the number of values, again **m**, to get the average, or mean, squared deviation.

σB2←1m∑i\=1m(xi−μB)2 \\sigma\_{B}^{2} \\leftarrow \\frac{1}{m}\\sum\_{i=1}^m (x\_i - \\mu\_B)^2 σB2​←m1​i\=1∑m​(xi​−μB​)2

## Normalizing output values

Once we have the mean and variance, we can use them to normalize the values with the following equation. For each value, it subtracts the mean and divides by the (almost) standard deviation. (You've probably heard of standard deviation many times, but if you have not studied statistics you might not know that the standard deviation is actually the square root of the mean squared deviation.)

xi^←xi−μBσB2+ϵ \\hat{x\_i} \\leftarrow \\frac{x\_i - \\mu\_B}{\\sqrt{\\sigma\_{B}^{2} + \\epsilon}} xi​^​←σB2​+ϵ ​xi​−μB​​

Above, we said "(almost) standard deviation". That's because the real standard deviation for the batch is calculated by σB2\\sqrt{\\sigma\_{B}^{2}}σB2​ ​

but the above formula adds the term epsilon before taking the square root. The epsilon can be any small, positive constant, ex. the value `0.001`. It is there partially to make sure we don't try to divide by zero, but it also acts to increase the variance slightly for each batch.

Why add this extra value and mimic an increase in variance? Statistically, this makes sense because even though we are normalizing one batch at a time, we are also trying to estimate the population distribution – the total training set, which itself an estimate of the larger population of inputs your network wants to handle. The variance of a population is _typically_ higher than the variance for any sample taken from that population, especially when you use a small sample size (a small sample is more likely to include values near the peak of a population distribution), so increasing the variance a little bit for each batch helps take that into account.

At this point, we have a normalized value, represented as xi^\\hat{x\_i}xi​^​

But rather than use it directly, we multiply it by a **gamma** value, and then add a **beta** value. Both gamma and beta are learnable parameters of the network and serve to scale and shift the normalized value, respectively. Because they are learnable just like weights, they give your network some extra knobs to tweak during training to help it learn the function it is trying to approximate.

yi←γxi^+β y\_i \\leftarrow \\gamma \\hat{x\_i} + \\beta yi​←γxi​^​+β

We now have the final batch-normalized output of our layer, which we would then pass to a non-linear activation function like sigmoid, tanh, ReLU, Leaky ReLU, etc. In the original batch normalization paper, they mention that there might be cases when you'd want to perform the batch normalization _after_ the non-linearity instead of before, but it is difficult to find any uses like that in practice.

#### Next, take a look at the effect of batch normalization, by applying it to a PyTorch model!

## Adding Batch Normalization Layers to a PyTorch Model

In the last notebook, you saw how a model with batch normalization applied reached a lower training loss and higher test accuracy! There are quite a few comments in that code, and I just want to recap a few of the most important lines.

To add batch normalization layers to a PyTorch model:

- You add batch normalization to layers inside the`__init__` function.
- Layers with batch normalization do not include a bias term. So, for linear or convolutional layers, you'll need to set `bias=False` if you plan to add batch normalization on the outputs.
- You can use PyTorch's \[BatchNorm1d\] function to handle the math on linear outputs _or_ \[BatchNorm2d\] for 2D outputs, like filtered images from convolutional layers.
- You add the batch normalization layer _before_ calling the activation function, so it always goes layer > batch norm > activation.

Finally, when you tested your model, you set it to `.eval()` mode, which ensures that the batch normalization layers use the _population_rather than the _batch_ mean and variance (as they do during training).

![](https://video.udacity-data.com/topher/2018/November/5be38916_screen-shot-2018-11-07-at-4.53.24-pm/screen-shot-2018-11-07-at-4.53.24-pm.png)

Batch normalization benefits

### The takeaway

By using batch normalization to normalize the inputs at each layer of a network, we can make these inputs more consistent and thus reduce oscillations that may happen in gradient descent calculations. This helps us build deeper models that also converge faster!

> Take a look at the [PyTorch BatchNorm2d documentation](https://pytorch.org/docs/stable/nn.html#batchnorm2d) to learn more about how to add batch normalization to a model, and how data is transformed during training (and evaluation).

# Benefits of Batch Normalization

Batch normalization optimizes network training. It has been shown to have several benefits:

1. **Networks train faster** – Each training _iteration_ will actually be slower because of the extra calculations during the forward pass and the additional hyperparameters to train during back propagation. However, it should converge much more quickly, so training should be faster overall.
2. **Allows higher learning rates** – Gradient descent usually requires small learning rates for the network to converge. And as networks get deeper, their gradients get smaller during back propagation so they require even more iterations. Using batch normalization allows us to use much higher learning rates, which further increases the speed at which networks train.
3. **Makes weights easier to initialize** – Weight initialization can be difficult, and it's even more difficult when creating deeper networks. Batch normalization seems to allow us to be much less careful about choosing our initial starting weights.
4. **Makes more activation functions viable** – Some activation functions do not work well in some situations. Sigmoids lose their gradient pretty quickly, which means they can't be used in deep networks. And ReLUs often die out during training, where they stop learning completely, so we need to be careful about the range of values fed into them. Because batch normalization regulates the values going into each activation function, non-linearlities that don't seem to work well in deep networks actually become viable again.
5. **Simplifies the creation of deeper networks** – Because of the first 4 items listed above, it is easier to build and faster to train deeper neural networks when using batch normalization. And it's been shown that deeper networks generally produce better results, so that's great.
6. **Provides a bit of regularization** – Batch normalization adds a little noise to your network. In some cases, such as in Inception modules, batch normalization has been shown to work as well as dropout. But in general, consider batch normalization as a bit of extra regularization, possibly allowing you to reduce some of the dropout you might add to a network.
7. **May give better results overall** – Some tests seem to show batch normalization actually improves the training results. However, it's really an optimization to help train faster, so you shouldn't think of it as a way to make your network better. But since it lets you train networks faster, that means you can iterate over more designs more quickly. It also lets you build deeper networks, which are usually better. So when you factor in everything, you're probably going to end up with better results if you build your networks with batch normalization.

