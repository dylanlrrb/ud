## Requires Changes

### 6 specifications require changes

Good submission! Well done!

The model design and execution are perfect. You have implemented a lot of interesting functions as well, like Running TSNE on extracted features.

But there are some small tweaks to be made to make it perfect. Please find the details in the respective sections.  
Also, you have to test with new images outside the given data. Please download some images from the internet, test, and resubmit. This gives us a better insight on how the model is working and if it is robust.  
Hope my suggestions are helpful and you enjoy the learning process.

Please don't hesitate to take help from Knowledge platform: [https://knowledge.udacity.com](https://knowledge.udacity.com/) regarding the same. It is an interactive platform where mentors will help you resolve technical issues. Hope you enjoy the learning process.

All the best! 👍 👍

## Files Submitted

The submission includes the required notebook file and HTML file. When the HTML file is created, all the code cells in the notebook need to have been run so that reviewers can see the final implementation and output.

All required files are submitted. Thank you.

## Step 1: Create a CNN to Classify Landmarks (from Scratch)

The submission randomly splits the images at `landmark_images/train` into train and validation sets. The submission then creates a data loader for the created train set, a data loader for the created validation set, and a data loader for the images at `landmark_images/test`.

`'valid': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, sampler=valid_sampler),`  
You cannot use test dataset for validation since it results in data leakage. It is a critical issue in testing.  
Validation set is used for determining the parameters of the model, and test set is used for evaluate the performance of the model in an unseen dataset. Hence the data from the test dataset should not be visible until the whole training is over.

Please use training set to split the validation dataset.

More info: [https://machinelearningmastery.com/data-leakage-machine-learning/](https://machinelearningmastery.com/data-leakage-machine-learning/)

Good use of SubsetRandomSampler. The images are shuffled using np.random.shuffle(indices).  
The PyTorch's DataLoader class is used correctly. Learn more about Dataloaders [here](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)  
Well done 👍

I suggest using 20% of the data for validation.

Answer describes each step of the image preprocessing and augmentation. Augmentation (cropping, rotating, etc.) is not a requirement.

_I created a validation dataset out of 50% of the test dataset, I chose to use the test dataset rather than the train dataset because after some research I found that I should be using un-augmented images for validataion_

Yes you should not use data augmentation in validation dataset but it is not a serious issue. It just hinders performance.  
But using test dataset for validation results in more serious issue of data leakage.  
Validation set is used for determining the parameters of the model, and test set is used for evaluate the performance of the model in an unseen dataset. Hence the data from the test dataset should not be visible until the whole training is over.

Please use training set to split the validation dataset.

In a small exercise like this it might not seem obvious but in an industry project or deployments, these issues can be critical.

The submission displays at least 5 images from the train data loader, and labels each image with its class name (e.g., "Golden Gate Bridge").

You have printed images with labels.  
Well done 👍. Good use of matplotlib library. We can see the transformed images.

The submission chooses appropriate loss and optimization functions for this classification task.

NLLLoss is the right loss function. 👍  
Another option is CrossEntropyLoss which already has built in softmax function.

Adam is more suitable for image classification problems. 👍  
Check out the comparative study on optimizers. [link](https://medium.com/syncedreview/iclr-2019-fast-as-adam-good-as-sgd-new-optimizer-has-both-78e37e8f9a34#:~:text=SGD%20is%20a%20variant%20of,random%20selection%20of%20data%20examples.&text=Essentially%20Adam%20is%20an%20algorithm,optimization%20of%20stochastic%20objective%20functions).

The submission specifies a CNN architecture.

- 4 layers of convolutional layers gives good enough depth for the model to learn details of the image. Well done.
- 16, 32, 64, 64 are enough filters to extract features.
- You have used stride 2 which reduces the size of the feature maps which is faster.
- Relu is a good activation to use in hidden layers as it prevents vanishing gradients.
- 4 fully connected layers give a good depth to the model. It helps to bring all the features together and make a big feature that gives a meaningful output.
- Good use of dropout layers. It regularizes the network.
- You are using LogSoftmax in combination with NLLloss which works fine.  
    The model is deep(many layers) and wide(many parameters) enough. Thats good. 👍👍  
    Read more about convolutional arithmetic here. [https://arxiv.org/abs/1603.07285](https://arxiv.org/abs/1603.07285)

## SUGGESTION:

You can use pooling layers instead of using stride 2 in convolutional layers. Pooling layers not only reduce the size of the feature maps they also help you to choose important features.  
Try the batch normalization for faster convergence, that is, the model learns faster. [https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/](https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/)

Answer describes the reasoning behind the selection of layer types.

_I created 4 convolutional layers and opted to not use any pooling layers, rather I used a stride of 2 and a padding of 1 so that each successive convolutional layer would be half the dimensions of the previous layer. I chose to double the number of feature maps at each convolutional layer, expecting that each successive layer would extract more numerous and higher level features from the layer below.  
Finally I flattened the last convolutional layer and used it as input to a multilayer perceptron classifier with 4 hidden layers, the last of which has an out feature for each of the 50 classes.  
Because I am using NLLLoss as the criterion I did a log\_softmax on the outputs of the final layer._

The details of the layers and the chosen parameters are articulated clearly. You have thought about how and why you made a certain decision for the Deep Learning model which is very important.

_Further reading:  
[https://cs231n.github.io/convolutional-networks/#conv](https://cs231n.github.io/convolutional-networks/#conv)_

The submission implements an algorithm to train a model for a number of epochs and save the "best" result.

Train algorithm implemented correctly. I see that you are saving the model whenever the validation loss decreases. Well done. 👍👍

The submission implements a custom weight initialization function that modifies all the weights of the model. The submission does not cause the training loss or validation loss to explode to `nan`.

The custom initialization is implemented correctly. Well done 👏👏

_Further Reading: [https://www.kdnuggets.com/2018/06/deep-learning-best-practices-weight-initialization.html](https://www.kdnuggets.com/2018/06/deep-learning-best-practices-weight-initialization.html)_

## SUGGESTION:

Please initialize the convolution layers as well.

The trained model attains at least 20% accuracy on the test set.

Good work...  
BUT  
Let's revisit this section after making the necessary changes to the validation dataset.

## Step 2: Create a CNN to Classify Landmarks (using Transfer Learning)

The submission specifies a model architecture that uses part of a pre-trained model.

AS mentioned earlier, you cannot use test dataset for validation since it results in data leakage. It is a critical issue in testing.  
Please use training set to split the validation dataset.  
Validation set is used for determining the parameters of the model, and test set is used for evaluate the performance of the model in an unseen dataset. Hence the data from the test dataset should not be visible until the whole training is over.  
More info: [https://machinelearningmastery.com/data-leakage-machine-learning/](https://machinelearningmastery.com/data-leakage-machine-learning/)

The submission details why the chosen architecture is suitable for this classification task.

The transfer learning is correctly implemented using pre-trained model, its layers are correctly frozen. The classifier is correctly replaced with new trainable FC layer. Well done 👍👍  
[https://pytorch.org/tutorials/beginner/transfer\_learning\_tutorial.html](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

The submission uses model checkpointing to train the model and saves the model weights with the best validation loss.

Train algorithm implemented correctly. I see that you are saving the model whenever the validation loss decreases. Well done.

Accuracy on the test set is 60% or greater.

Good work...  
BUT  
Let's revisit this section after making the necessary changes to the validation dataset.

## Step 3: Write Your Landmark Prediction Algorithm

The submission implements functionality to use the transfer learned CNN from Step 2 to predict top k landmarks. The returned predictions are the names of the landmarks (e.g., "Golden Gate Bridge").

The predict\_landmarks pre-processes the given image correctly.  
The model is called in eval mode.  
It returns the class after passing it through the model.  
It predicts top k landmarks. Good use of torch.exp.  
Well done. 👍 👍  
_Further reading:  
[https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch](https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch)_

The submission displays a given image and uses the functionality in "Write Your Algorithm, Part 1" to predict the top 3 landmarks.

Your algorithm prints the top 3 predictions of the input landmark.  
Correct results. 👍👍

The submission tests at least 4 images.

You have to also test with new images outside the given data. Please download some images from the internet, test, and resubmit. This gives us a better insight on how the model is working and if it is robust.

Submission provides at least three possible points of improvement for the classification algorithm.

_training on a larger dataset with more and varied perspectives on various landmarks_  
**Yes increasing the data with diversity is the best way for the model to learn better.**

_experimenting with different pretrained models for transfer learning; my model uses VGG but maybe another base architecture would perform better (e.g. Resnet, Alexnet, etc.)_  
**Do try other architectures like ResNet, DenseNet and compare the results.**

_not freezing the layers from the pretrained model and allowing the training to fine tune the earlier layers_  
**Yes you can fine tune more pre-trained layers by un-freezing them**