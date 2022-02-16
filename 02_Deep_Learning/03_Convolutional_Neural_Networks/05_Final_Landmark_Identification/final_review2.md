## Meets Specifications

Congratulations, the project is complete.

Couple of resources for more on the subject if interested:  
[Ensembles](https://machinelearningmastery.com/ensemble-methods-for-deep-learning-neural-networks/)  
[GANS](https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/)  
Hope you found the material interesting and enjoyable.  
Great job and good luck going forward.

## Files Submitted

The submission includes the required notebook file and HTML file. When the HTML file is created, all the code cells in the notebook need to have been run so that reviewers can see the final implementation and output.

## Step 1: Create a CNN to Classify Landmarks (from Scratch)

The submission randomly splits the images at `landmark_images/train` into train and validation sets. The submission then creates a data loader for the created train set, a data loader for the created validation set, and a data loader for the images at `landmark_images/test`.

Random splitting of train data and data loader creation of all 3 which are required is correct.

Answer describes each step of the image preprocessing and augmentation. Augmentation (cropping, rotating, etc.) is not a requirement.

Augmentation applied properly.  
Good using the ImageNet normalization values.  
Answer reflects data loader creation and choices are reasonable.

The submission displays at least 5 images from the train data loader, and labels each image with its class name (e.g., "Golden Gate Bridge").

The submission chooses appropriate loss and optimization functions for this classification task.

The submission specifies a CNN architecture.

Answer describes the reasoning behind the selection of layer types.

The submission implements an algorithm to train a model for a number of epochs and save the "best" result.

The submission implements a custom weight initialization function that modifies all the weights of the model. The submission does not cause the training loss or validation loss to explode to `nan`.

The trained model attains at least 20% accuracy on the test set.

Excellent % for a scratch model.  
Might try a stride > 1 for one or more convolutional layers for improved performance if you work on a similar problem in the future.

## Step 2: Create a CNN to Classify Landmarks (using Transfer Learning)

The submission specifies a model architecture that uses part of a pre-trained model.

Good you copied over the correct data loaders.  
Even shorter one liner for this

```
loaders_transfer = loaders_scratch.copy()
```

Your thoughtful multi layer change may not perform better than merely replacing the final layer for this particular  
task of transfer learning(though it might).  
This task is not so complex and the power of transfer learning is extensive for it thus a deep addition may not be  
optimal. Somewhat counterintuitive, perhaps, but similar to overfitting with an overly wide and/or deep architecture.

Also, using NLLLoss() and log\_softmax is technically correct but it is standard to use CrossEntropyLoss() which handles  
the output and then no activation is required for the final layer output. Optimized by Pytorch under the hood. Probably makes not difference in the output in this case just a note for general use.  
[Pytorch CNN](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

The submission details why the chosen architecture is suitable for this classification task.

The submission uses model checkpointing to train the model and saves the model weights with the best validation loss.

Accuracy on the test set is 60% or greater.

VGG16 does the job > 60% accuracy.

## Step 3: Write Your Landmark Prediction Algorithm

The submission implements functionality to use the transfer learned CNN from Step 2 to predict top k landmarks. The returned predictions are the names of the landmarks (e.g., "Golden Gate Bridge").

The submission displays a given image and uses the functionality in "Write Your Algorithm, Part 1" to predict the top 3 landmarks.

The submission tests at least 4 images.

Output format looks correc. Proper set of test images.  
Nice added functionality.

Just a note that with test images in their own folder(or however you might access them alone) you could loop through it and call your function once. Less typing and/or copy/paste and somewhat cleaner code.

Submission provides at least three possible points of improvement for the classification algorithm.