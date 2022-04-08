### DCGAN Paper

It's always good to take a look at the original paper when you can. Many papers discuss both the theory and training details of deep learning networks, and you can read the DCGAN paper, [Unsupervised Representational Learning with Deep Convolutional Generative Adversarial Networks, at this link](https://arxiv.org/pdf/1511.06434.pdf). I especially like the section they have on model architectures, which is pasted for convenience as an image, below.

![](https://video.udacity-data.com/topher/2018/November/5be79687_screen-shot-2018-11-10-at-6.39.49-pm/screen-shot-2018-11-10-at-6.39.49-pm.png)

Architecture guidelines for stable DCGANs.

Why no bias?
The reason there is no bias for our convolutional layers is because we have batch normalization applied to their outputs. The goal of batch normalization is to get outputs with:

mean = 0
standard deviation = 1
Since we want the mean to be 0, we do not want to add an offset (bias) that will deviate from 0. We want the outputs of our convolutional layer to rely only on the coefficient weights.

### Forgotten: Scaling Step!

In the above training code, I forgot to include the scaling function for our real training images! Before passing in the images to our discriminator there should be a scaling step:

```
# important rescaling step
real_images = scale(real_images)
```

This has been fixed in the exercise and solution notebooks in the classroom and the Github repository.

**You should see a smoother decrease in Generator loss** (as pictured below) with the addition of this line of code _and_ slightly different-looking generated, sample images.

![](https://video.udacity-data.com/topher/2018/November/5bea0fba_screen-shot-2018-11-12-at-3.41.20-pm/screen-shot-2018-11-12-at-3.41.20-pm.png)

Actual training loss

In hindsight, I should have known that something was wrong with my training process due to the increasing and slightly unstable Generator loss in this video! For me, this is a good lesson in using a mix of intuition and training results to double check my work.

The correct scaling code and solution has been fixed in the in-classroom code and in [the Github repo](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/dcgan-svhn/DCGAN_Solution.ipynb).

## GANs for Illuminating Model Weaknesses

GANs are not only used for image generation, they are also used to find weaknesses in existing, trained models. The adversarial examples that a generator learns to make, can be designed to _trick_ a pre-trained model. Essentially, small perturbations in images can cause a classifier (like AlexNet or a known image classifier) to fail pretty spectacularly!

> [This OpenAI blog post](https://blog.openai.com/adversarial-example-research/) details how adversarial examples can be used to "attack" existing models, and discusses potential security issues. And one example of a perturbation that causes misclassification can be seen below.

![Image of a panda misclassified as an ape.](https://video.udacity-data.com/topher/2018/November/5bea117c_screen-shot-2018-11-12-at-3.47.20-pm/screen-shot-2018-11-12-at-3.47.20-pm.png)

Adding a small amount of noise to an image of a panda causes a model to misclassify it as a [gibbon](https://en.wikipedia.org/wiki/Gibbon), which is a kind of ape. One of the interesting parts of this is the model's confidence. With this noise it is **99.3**% confident that this is an image of a gibbon, when we can pretty clearly see that it is a panda!

## Other Interesting Applications of GANs

So far, you've seen a lot of examples of how GANs might be used for image generation and transformation. GANs are a relatively new formulation and so there are some really exciting research directions that include GANs. I didn't have time to cover them all in video, so I wanted to highlight a few of my favorite examples, here, and link to some resources that I've found helpful! **This page is for those who are interested in learning more about GANs and curious to learn about semi-supervised learning.**

### 1\. Semi-Supervised Learning

Semi-supervised models are used when you only have a _few_ labeled data points. The motivation for this kind of model is that, we increasingly have a lot of raw data, and the task of labelling data is tedious, time-consuming, and often, sensitive to human error. Semi-supervised models give us a way to learn from a large set of data with only a few labels, and they perform surprisingly well even though the amount of labeled data you have is relatively tiny. Ian Goodfellow has put together a video on this top, which you can see, below.

### Semi-Supervised Learning in PyTorch

There is a readable implementation of a semi-supervised GAN in [this Github repository](https://github.com/Sleepychord/ImprovedGAN-pytorch). If you'd like to implement this in code, I suggest reading through that code!

### 2\. Domain Invariance

Consider [this car classification example](https://arxiv.org/abs/1709.02480). From the abstract, researchers (Timnit Gebru, et. al) wanted to:

> develop a computer vision pipeline to predict income, per capita carbon emission, crime rates and other city attributes from a single source of publicly available visual data. We first detect cars in 50 million images across 200 of the largest US cities and train a model to predict demographic attributes using the detected cars. To facilitate our work, we have collected the largest and most challenging fine-grained dataset reported to date consisting of over 2600 classes of cars comprised of images from Google Street View and other web sources, classified by car experts to account for even the most subtle of visual differences.

One interesting thing to note is that these researchers obtained some manually-labeled Streetview data _and_ data from other sources. I'll call these image sources, domains. So Streetview is a domain and another source, say cars.com is separate domain.

![](https://video.udacity-data.com/topher/2018/November/5beb5911_screen-shot-2018-11-13-at-3.06.36-pm/screen-shot-2018-11-13-at-3.06.36-pm.png)

Different image sources for the paper, [Fine-Grained Car Detection for Visual Census Estimation](https://arxiv.org/abs/1709.02480)

The researchers then had to find a way to combine what they learned from these multiple sources! They did this with the use of multiple classifiers; adversarial networks that do _not_ include a Generator, just two classifiers.

> - One classifier is learning to recognize car types
> - And another is learning to classify whether a car image came from Google Streetview _or_ cars.com, given the extracted features from that image

So, the first classier’s job is to classify the car image correctly _and_ to **trick the second classifier** so that the second classifier cannot tell whether the extracted image features indicate an image from the Streetview or cars.com domain!

The idea is: if the second classifier cannot tell which domain the features are from, then this indicates that these features are shared among the two domains, and you’ve found features that are **domain-invariant**.

Domain-invariance can be applied to a number of applications in which you want to find features that are invariant between two different domains. These can be image domains or domains based on different population demographics and so on. This is also sometimes referred to as [adversarial feature learning](https://arxiv.org/pdf/1705.11122.pdf).

### 3\. Ethical and Artistic Applications: Further Reading

- [Ethical implications of GANs](https://www.newyorker.com/magazine/2018/11/12/in-the-age-of-ai-is-seeing-still-believing) and when "fake" images can give us information about reality.
- [Do Androids Dream in Balenciaga?](https://www.ssense.com/en-us/editorial/fashion/do-androids-dream-of-balenciaga-ss29) note that the author briefly talks about generative models having artistic potential rather than ethical implications, but the two go hand in hand. The generator, in this case, will recreate what it sees on the fashion runway; typically thin, white bodies that do not represent the diversity of people in the world (or even the diversity of people who buy Balenciaga).

