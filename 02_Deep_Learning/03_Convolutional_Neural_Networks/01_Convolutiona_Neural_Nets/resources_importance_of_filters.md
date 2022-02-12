# The Importance of Filters

What you've just learned about different types of filters will be really important as you progress through this course, especially when you get to Convolutional Neural Networks (CNNs). CNNs are a kind of deep learning model that can learn to do things like image classification and object recognition. They keep track of spatial information and _learn_ to extract features like the edges of objects in something called a **convolutional layer**. Below you'll see an simple CNN structure, made of multiple layers, below, including this "convolutional layer".

![](https://video.udacity-data.com/topher/2018/May/5b1070e4_screen-shot-2018-05-31-at-2.59.36-pm/screen-shot-2018-05-31-at-2.59.36-pm.png)

Layers in a CNN.

## Convolutional Layer

The convolutional layer is produced by applying a series of many different image filters, also known as convolutional kernels, to an input image.

![](https://video.udacity-data.com/topher/2018/May/5b10723a_screen-shot-2018-05-31-at-3.06.07-pm/screen-shot-2018-05-31-at-3.06.07-pm.png)

4 kernels = 4 filtered images.

In the example shown, 4 different filters produce 4 differently filtered output images. When we stack these images, we form a complete convolutional layer with a depth of 4!

![](https://video.udacity-data.com/topher/2018/May/5b10729b_screen-shot-2018-05-31-at-3.07.03-pm/screen-shot-2018-05-31-at-3.07.03-pm.png)

A convolutional layer.

## Learning

In the code you've been working with, you've been setting the values of filter weights explicitly, but neural networks will actually _learn_ the best filter weights as they train on a set of image data. You'll learn all about this type of neural network later in this section, but know that high-pass and low-pass filters are what define the behavior of a network like this, and you know how to code those from scratch!

In practice, you'll also find that many neural networks learn to detect the edges of images because the edges of object contain valuable information about the shape of an object.