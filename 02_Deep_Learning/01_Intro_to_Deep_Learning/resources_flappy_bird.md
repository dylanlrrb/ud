### Flappy Bird

In this example, you'll get to see a deep learning agent playing Flappy Bird! You have the option to train the agent yourself, but for now let's just start with the pre-trained network given by the author. Note that the following agent is able to play without being told any information about the structure of the game or its rules. It automatically discovers the rules of the game by finding out how it did on each iteration.

We will be following [this repository](https://github.com/yenchenlin/DeepLearningFlappyBird) by Yenchen Lin.(#idea)

![](https://video.udacity-data.com/topher/2017/January/588acc5c_flappy-bird/flappy-bird.jpg)

### Instructions

1. Install miniconda or anaconda if you have not already.
2. Create an environment for flappybird
    - Mac/Linux: `conda create --name=flappybird python=2.7`
    - Windows: `conda create --name=flappybird python=3.5`
3. Enter your conda environment: `conda activate flappybird`
4. `conda install opencv`
    - If you encounter an error here, you may try an **alternate** download path and _instead_ type `conda install --channel https://conda.anaconda.org/menpo opencv3`
5. `pip install pygame`
6. `pip install tensorflow==0.12`
7. `git clone https://github.com/yenchenlin/DeepLearningFlappyBird.git`
    - If you don't have git installed, you can download and extract the zip archive directly from [the repository](https://github.com/yenchenlin/DeepLearningFlappyBird)
8. `cd DeepLearningFlappyBird`
    - If you downloaded the archive, you will need to navigate to the extracted folder **DeepLearningFlappyBird-master** instead
9. `python deep_q_network.py`

If all went correctly, you should be seeing a deep learning based agent play Flappy Bird! The repository contains instructions for training your own agent if you're interested!

You can also, typically, force-quit out of the game by returning to your terminal window and typing `Command or Ctrl + C`.

[demystifying-deep-reinforcement-learning](https://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/)