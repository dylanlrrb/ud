# Congratulations

## Great submission!

### All functions were implemented correctly, and the final algorithm seems to work quite well.

### Suggestions to make your project better:

- Create a deeper model and use it to generate larger (say 128x128) images of faces.
- Implement a learning rate that evolves over time as they did in this [CycleGAN Github repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
- See if you can extend this model and use a CycleGAN to learn to swap different kinds of faces. For example, learn a mapping between faces that have and do not have eye/lip makeup, as they did [in this paper](https://gfx.cs.princeton.edu/pubs/Chang_2018_PAS/Chang-CVPR-2018.pdf).

### Some additional materials related to topics discussed in this course.

- [How to to select the batch\_size vs the number of epochs](https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network)
- [Google AI Blog](https://research.googleblog.com/2017/12/tfgan-lightweight-library-for.html)

## Required Files and Tests

The project submission contains the project notebook, called “dlnd\_face\_generation.ipynb”.

## Well done!

### All required files were included

All the unit tests in project have passed.

## All the unit tests in project have passed

[![faces_UNITESTS.png](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/147572/1649542141/faces_UNITESTS.png)](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/147572/1649542141/faces_UNITESTS.png)

## Great work.

### Unit testing is used to ensure that your code is free from all bugs without getting confused with the interactions with all the other code.

### But unit tests cannot catch every issue. So your code can have bugs and even though unit tests would pass.

### You can read more about unit testing in [Testing Your Code](https://docs.python-guide.org/writing/tests/)

## Data Loading and Processing

The function `get_dataloader` should transform image data into resized, Tensor image types and return a DataLoader that batches all the training data into an appropriate size.

## Dataloader is essential to PyTorch.

### The **get\_dataloader()** function has been implemented correctly. Well done!

### All images were resized to image\_size=32.

### There are two main steps to completing this dataloader function:

1. Create a transformed dataset. This is typically done with a call to [PyTorch's ImageFolder wrapper](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder).
2. Creating and returning a Dataloader that batches (and optionally shuffles) the data.

Pre-process the images by creating a `scale` function that scales images into a given pixel range. This function should be used later, in the training loop.

## Good work normalizing inputs using the **scale()** function.

### The **scale()** function scaled images into a pixel range of -1 to 1.

## Build the Adversarial Networks

The Discriminator class is implemented correctly; it outputs one value that will determine whether an image is real or fake.

## Overall you did a fine job implementing the Discriminator as a simple convolution network.

[![Captura de tela de 2022-04-09 19-07-19.png](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/147572/1649542189/Captura_de_tela_de_2022-04-09_19-07-19.png)](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/147572/1649542189/Captura_de_tela_de_2022-04-09_19-07-19.png)

### Comments about your code:

- You chose not to use pooling layers to decrease the spatial size. Max pooling generates sparse gradients, which affects the stability of GAN training.
- You correctly used ReLU activations to introduce non-linearity and to allow gradients to flow backwards through the layer unimpeded.
- You have used Batch normalization to transform the input to zero mean/unit variance distributions.

### Batch normalization is becoming very popular to further improve the performance of the model.

### Batch normalization layers avoid covariate shift and accelerate the training process

### Look at these suggested readings:

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)
- [Understanding the backward pass through Batch Normalization Layer](http://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)

The Generator class is implemented correctly; it outputs an image of the same shape as the processed training data.

## Good work! The implementation of the generator to generate an image using z was correctly coded.

[![Captura de tela de 2022-04-09 19-07-25.png](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/147572/1649542217/Captura_de_tela_de_2022-04-09_19-07-25.png)](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/147572/1649542217/Captura_de_tela_de_2022-04-09_19-07-25.png)

### This function should be able to reuse the variables in the neural network.

### Comments about your code:

- Most of the suggestions are same for both Generator and Discriminator.
- Tanh as the last layer of the generator output. This means that we'll have to normalize the input images to be between -1 and 1.

This function should initialize the weights of any convolutional or linear layer with weights taken from a normal distribution with a mean = 0 and standard deviation = 0.02.

## Well done!

### You have initialized correctly using normal distribution.

## Optimization Strategy

The loss functions take in the outputs from a discriminator and return the real or fake loss.

## Well done getting the real and fake loss.

There are optimizers for updating the weights of the discriminator and generator. These optimizers should have appropriate hyperparameters.

## Given your network architecture, the choice of hyper-parameter are reasonable.

### Comments about you code:

- Good choice using **Adam optimizers** for updating the weights of the discriminator and generator.

### Here Sebastian Ruder explains [Adam optimizers](https://ruder.io/optimizing-gradient-descent/index.html#adam) a little bit.

## Training and Results

Real training images should be scaled appropriately. The training loop should alternate between training the discriminator and generator networks.

## Good job utilizing the GPU for all model inputs!

### Real training images were scaled appropriately.

### The training loop alternated between training the discriminator and generator networks.

### Soumith Chintala is one of the author of original DCGAN paper. Here he guides how to train:

- [How to Train a GAN? Tips and tricks to make GANs work](https://github.com/soumith/ganhacks).
- [Workshop on Adversarial Training - How to train a GAN](https://www.youtube.com/watch?v=X1mUN6dD8uE)

There is not an exact answer here, but the models should be deep enough to recognize facial features and the optimizers should have parameters that help wth model convergence.

## Good work!

### The models were deep enough to recognize facial features and give good predictions.

[![Captura de tela de 2022-04-09 19-07-36.png](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/147572/1649542274/Captura_de_tela_de_2022-04-09_19-07-36.png)](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/147572/1649542274/Captura_de_tela_de_2022-04-09_19-07-36.png)

### The parameters for the optimizers were appropriate for convergence.

The project generates realistic faces. It should be obvious that generated sample images look like faces.

## Good job!

### It was generated almost realistic faces but it looks like human faces.

[![Captura de tela de 2022-04-09 19-07-43.png](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/147572/1649542307/Captura_de_tela_de_2022-04-09_19-07-43.png)](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/147572/1649542307/Captura_de_tela_de_2022-04-09_19-07-43.png)

### This confirms the robustness of your models.

The **question** about model improvement is answered.

[![Captura de tela de 2022-04-09 19-07-54.png](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/147572/1649542334/Captura_de_tela_de_2022-04-09_19-07-54.png)](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/147572/1649542334/Captura_de_tela_de_2022-04-09_19-07-54.png)

## Nice work describing the reasoning behind the selection of layer types in question 4.