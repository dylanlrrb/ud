## Links to Resources

- [StackGAN](https://arxiv.org/abs/1612.03242) realistic image synthesis
- [iGAN](https://github.com/junyanz/iGAN) interactive image generation
- CartoonGAN, linked below

You'll learn much more about Pix2Pix and CycleGAN formulations, later in this program!

#### Supporting Materials

- [CartoonGAN](https://video.udacity-data.com/topher/2018/November/5bea23cd_cartoongan/cartoongan.pdf)

You can find more information on the graph in the video in Figure 1 of [https://arxiv.org/pdf/1406.2661.pdf](https://arxiv.org/pdf/1406.2661.pdf).

### Improved Training Techniques for GANs

Read the paper, linked below, that describes improved training techniques for GANs!

#### Supporting Materials

- [Improved-training-techniques](https://video.udacity-data.com/topher/2018/November/5bea0c6a_improved-training-techniques/improved-training-techniques.pdf)

### GAN examples

If you'd like to read about even more applications of GANs, I recommend [this Medium article](https://medium.com/@jonathan_hui/gan-some-cool-applications-of-gans-4c9ecca35900) which does an overview of interesting applications!

The tulip generation model was created by the artist Anna Ridler, and you can read about her data collection method and inspiration in [this article](https://www.fastcompany.com/90237233/this-ai-dreams-in-tulips). Also, check out the [full-length video](https://vimeo.com/287645190)!

### The universal approximation function

The universal approximation theorem states that a feed-forward network with a **single** hidden layer is able to approximate certain continuous functions. A few assumptions are made about the functions operating on a subset of real numbers and about the activation function applied to the output of this single layer. But this is very exciting! This theorem is saying that a simple, one-layer neural network can represent a wide variety of interesting functions. You can learn more about the theorem [here](https://en.wikipedia.org/wiki/Universal_approximation_theorem).

## Binary Cross Entropy Loss

We've mostly used plain cross entrpy loss in this program, which is a negative log loss applied to the output of a softmax layer. For a binary classification problem, as in _real_ or _fake_ image data, we can calculate the **binary cross entropy loss** as:

−\[ylog(y^)+(1−y)log(1−y^)\] -\[y\\log(\\hat{y}) +(1-y) \\log (1-\\hat{y})\] −\[ylog(y^​)+(1−y)log(1−y^​)\]

In other words, a sum of two log losses!

You can read the [PyTorch documentation, here](https://pytorch.org/docs/stable/nn.html#bceloss).

### eval

Note that the Generator should be set to `eval` mode for sample generation. It doesn't make too big a difference in this case, but this is to account for the different behavior that a **dropout** layer has during training vs during model evaluation.

So, in the workspace and Github repository code, we've added the correct evaluation code for generating samples, writing the line `G.eval()` before we generate samples. We strive to always keep the code that you'll be working with correct and up-to-date!


