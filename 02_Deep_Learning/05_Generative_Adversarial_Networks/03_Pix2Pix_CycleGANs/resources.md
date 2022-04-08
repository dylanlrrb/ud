Check out [MIT CSAIL's website](https://www.csail.mit.edu/) to look at the variety of research that is happening in this lab.

### Links to Related Work

- Ian Goodfellow's [original paper on GANs](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
- Face swap with [CycleGAN Face-off](https://arxiv.org/pdf/1712.03451.pdf)

### Objective Loss Functions

An objective function is typically a loss function that you seek to minimize (or in some cases maximize) during training a neural network. These are often expressed as a function that measures the difference between a prediction **y\_hat** and a true target **y**.

L(y,y^) \\mathcal{L} (y, \\hat{y}) L(y,y^​)

The objective function we've used the most in this program is [cross entropy loss](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html), which is a negative log loss applied to the output of a softmax layer. For a binary classification problem, as in _real_ or _fake_ image data, we can calculate the **binary cross entropy loss** as:

−\[ylog(y^)+(1−y)log(1−y^)\] -\[y\\log(\\hat{y}) +(1-y) \\log (1-\\hat{y})\] −\[ylog(y^​)+(1−y)log(1−y^​)\]

In other words, a sum of two log losses!

* * *

In the notation in the _next_ video, you'll see that **y\_hat** is the output of the discriminator; our predicted class.

### Latent Space

Latent means "hidden" or "concealed". In the context of neural networks, a latent space often means a feature space, and **a latent vector is just a compressed, feature-level representation of an image**!

For example, when you created a simple autoencoder, the outputs that connected the encoder and decoder portion of a network made up a compressed representation that could also be referred to as a latent vector.

You can read more about latent space in \[this blog post\] as well as an interesting property of this space: recall that we can mathematically operate on vectors in vector space and with latent vectors, we can perform a kind of feature-level transformation on an image!

> This manipulation of latent space has even been used to create an [interactive GAN, iGAN](https://github.com/junyanz/iGAN/blob/master/README.md) for interactive image generation! I recommend reading the paper, linked in the Github readme.

## Pix2Pix resources

If you're interested in learning more, take a look at the [original Pix2Pix paper](https://arxiv.org/pdf/1611.07004.pdf). I'd also recommend this related work on creating high-res images: [high resolution, conditional GANs](https://tcwang0509.github.io/pix2pixHD/).

## Edges to Cats Demo

Try out Christopher Hesse's [image-to-image demo](https://affinelayer.com/pixsrv/) to get some really interesting (and sometimes creepy) results!

Many of these image are collected in the [Pix2Pix and CycleGAN Github repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) developed by Jun-Yan.

And you can read the [CycleGAN paper, here](https://arxiv.org/pdf/1703.10593.pdf).

### Importance of Cycle Consistency

A really interesting place to check cycle consistency is in language translation. Ideally, when you translate one word or phrase from, say, English to Spanish, if you translate it _back_ (from Spanish to English) you will get the same thing!

In fact, if you are interested in natural language processing, I suggest you look into this as an area of research; even Google Translate has a tough time with this. In fact, as an exercise, I want you to see if Google Translate passes the following cycle consistency test.

![](https://video.udacity-data.com/topher/2018/November/5bea15ac_screen-shot-2018-11-12-at-4.05.54-pm/screen-shot-2018-11-12-at-4.05.54-pm.png)

aroma, english to spanish

## Model Shortcomings

As with any new formulation, it's important not only to learn about its strengths and capabilities, but also, its weaknesses. A CycleGAN has a few shortcomings:

- It will only show one version of a transformed output even if there are multiple, possible outputs.
- A simple CycleGAN produces low-resolution images, though there is some research around [high-resolution GANs](https://github.com/NVIDIA/pix2pixHD)
- It occasionally fails! (One such case is pictured below.)

![](https://video.udacity-data.com/topher/2018/November/5be7a5ec_screen-shot-2018-11-10-at-7.45.35-pm/screen-shot-2018-11-10-at-7.45.35-pm.png)

A horses to zebra transformation.

## Resources

- [Augmented CycleGAN](https://arxiv.org/abs/1802.10151)
- Implementation of [StarGAN](https://github.com/yunjey/StarGAN)

