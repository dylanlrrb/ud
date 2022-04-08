### Residual Blocks

So far, we've mostly been defining networks as layers that are connected, one-by-one, _in sequence_, but there are a few other types of connections we can make! The connection that **residual blocks** make is sometimes called a skip connection. By summing up the output of one layer with the input of a previous layer, we are effectively making a connection between layers that are _not_ in sequence; we are _skipping_ over at least one layer with such a connection, as is indicated by the loop arrow below.

![](https://video.udacity-data.com/topher/2018/November/5bea1ed4_resnet-block/resnet-block.png)

Residual block with a skip connection between an input x and an output.

If you'd like to learn more about residual blocks and especially their effect on ResNet image classification models, I suggest reading [this blog post](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035), which details how ResNet (and its variants) work!

## Skip Connections

More generally, skip connections can be made between _several_ layers to combine the inputs of, say, a _much_ earlier layer and a later layer. These connections have been shown to be especially important in image segmentation tasks, in which you need to preserve spatial information over time (even when your input has gone through strided convolutional or pooling layers). One such example, is in [this paper](https://arxiv.org/abs/1608.04117) on skip connections and their role in medical image segmentation.

## LSGANs

Least squares can partly address the vanishing gradient problem for training deep GANs. The problem is as follows: for negative log-likelihood loss, when an input x is quite big, the gradient can get close to zero and become meaningless for training purposes. However, with a squared loss term, the gradient will actually increase with a larger x, as shown below.

![](https://video.udacity-data.com/topher/2018/November/5be7a428_screen-shot-2018-11-10-at-7.38.03-pm/screen-shot-2018-11-10-at-7.38.03-pm.png)

Loss patterns for large `x` values. Image from the [LSGAN paper](https://arxiv.org/abs/1611.04076).

Least square loss is just one variant of a GAN loss. There are many more variants such as a [Wasserstein GAN loss](https://arxiv.org/abs/1701.07875) and others. These loss variants sometimes can help stabilize training and produce better results. As you write your own code, you're encouraged to hypothesize, try out different loss functions, and see which works best in your case!

### Image Synthesis

Digital image transformation (say from summer to winter transformation) is still the domain of highly-skilled programmers and special-effects artists. However, with technology like trained GANs, image transformation and creation could become a tool for any photographer! As in this example, GANs can be trained to produce realistic images on a vast scale, and it will be interesting to see how image synthesis evolves as an art form and in machine learning.

