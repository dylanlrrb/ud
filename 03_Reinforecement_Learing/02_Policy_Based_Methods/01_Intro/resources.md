# More on the Policy

In the previous video, you learned how the agent could use a simple neural network architecture to approximate a **stochastic policy**. The agent passes the current environment state as input to the network, which returns action probabilities. Then, the agent samples from those probabilities to select an action.

![Neural network that encodes action probabilities](https://video.udacity-data.com/topher/2018/July/5b38f93a_screen-shot-2018-07-01-at-10.54.05-am/screen-shot-2018-07-01-at-10.54.05-am.png)

Neural network that encodes action probabilities ([Source](https://blog.openai.com/evolution-strategies/))

The same neural network architecture can be used to approximate a **deterministic policy**. Instead of sampling from the action probabilities, the agent need only choose the greedy action.


## What about continuous action spaces?

* * *

The CartPole environment has a discrete action space. So, how do we use a neural network to approximate a policy, if the environment has a continuous action space?

As you learned above, in the case of **_discrete_** action spaces, the neural network has one node for each possible action.

For **_continuous_** action spaces, the neural network has one node for each action entry (or index). For example, consider the action space of the [bipedal walker](https://github.com/openai/gym/wiki/BipedalWalker-v2) environment, shown in the figure below.

![Action space of `BipedalWalker-v2`](https://video.udacity-data.com/topher/2018/July/5b3901fa_screen-shot-2018-07-01-at-11.28.57-am/screen-shot-2018-07-01-at-11.28.57-am.png)

Action space of `BipedalWalker-v2` ([Source](https://github.com/openai/gym/wiki/BipedalWalker-v2))

In this case, any action is a vector of four numbers, so the output layer of the policy network will have four nodes.

Since every entry in the action must be a number between -1 and 1, we will add a [tanh activation function](https://pytorch.org/docs/stable/nn.html#torch.nn.Tanh) to the output layer.

As another example, consider the [continuous mountain car](https://github.com/openai/gym/wiki/MountainCarContinuous-v0) benchmark. The action space is shown in the figure below. Note that for this environment, the action must be a value between -1 and 1.

![Action space of `MountainCarContinuous-v0`](https://video.udacity-data.com/topher/2018/July/5b38ff31_screen-shot-2018-07-01-at-11.19.22-am/screen-shot-2018-07-01-at-11.19.22-am.png)

Action space of `MountainCarContinuous-v0` ([Source](https://github.com/openai/gym/wiki/MountainCarContinuous-v0))



## Gradient Ascent

* * *

**Gradient ascent** is similar to gradient descent.

- Gradient descent steps in the **_direction opposite the gradient_**, since it wants to minimize a function.
- Gradient ascent is otherwise identical, except we step in the **_direction of the gradient_**, to reach the maximum.

While we won't cover gradient-based methods in this lesson, you'll explore them later in the course!

## Local Minima

* * *

In the video above, you learned that **hill climbing** is a relatively simple algorithm that the agent can use to gradually improve the weights θ\\thetaθ in its policy network while interacting with the environment.

Note, however, that it's **_not_** guaranteed to always yield the weights of the optimal policy. This is because we can easily get stuck in a local maximum. In this lesson, you'll learn about some policy-based methods that are less prone to this.

## Additional Note

* * *

Note that [hill climbing](https://en.wikipedia.org/wiki/Hill_climbing) is not just for reinforcement learning! It is a general optimization method that is used to find the maximum of a function.


# Hill Climbing Pseudocode

## What's the difference between GGG and JJJ?

* * *

You might be wondering: what's the difference between the return that the agent collects in a single episode (GGG, _from the pseudocode above_) and the expected return JJJ?

Well ... in reinforcement learning, the goal of the agent is to find the value of the policy network weights θ\\thetaθ that maximizes **_expected_** return, which we have denoted by JJJ.

In the hill climbing algorithm, the values of θ\\thetaθ are evaluated according to how much return GGG they collected in a **_single episode_**. To see that this might be a little bit strange, note that due to randomness in the environment (and the policy, if it is stochastic), it is highly likely that if we collect a second episode with the same values for θ\\thetaθ, we'll likely get a different value for the return GGG. Because of this, the (sampled) return GGG is not a perfect estimate for the expected return JJJ, but it often turns out to be **_good enough_** in practice.


# Beyond Hill Climbing

In the previous video, you learned about the hill climbing algorithm.

We denoted the expected return by JJJ. Likewise, we used θ\\thetaθ to refer to the weights in the policy network. Then, since θ\\thetaθ encodes the policy, which influences how much reward the agent will likely receive, we know that JJJ is a function of θ\\thetaθ.

Despite the fact that we have no idea what that function J\=J(θ)J = J(\\theta)J\=J(θ) looks like, the _hill climbing_ algorithm helps us determine the value of θ\\thetaθ that maximizes it. Watch the video below to learn about some improvements you can make to the hill climbing algorithm!

_Note_: We refer to the general class of approaches that find argmaxθJ(θ)\\arg\\max\_{\\theta}J(\\theta)argmaxθ​J(θ) through randomly perturbing the most recent best estimate as **stochastic policy search**. Likewise, we can refer to JJJ as an **objective function**, which just refers to the fact that we'd like to _maximize_ it!


# More Black-Box Optimization

All of the algorithms that you’ve learned about in this lesson can be classified as **black-box optimization** techniques.

**Black-box** refers to the fact that in order to find the value of θ\\thetaθ that maximizes the function J\=J(θ)J = J(\\theta)J\=J(θ), we need only be able to estimate the value of JJJ at any potential value of θ\\thetaθ.

That is, both hill climbing and steepest ascent hill climbing don't know that we're solving a reinforcement learning problem, and they do not care that the function we're trying to maximize corresponds to the expected return.

These algorithms only know that for each value of θ\\thetaθ, there's a corresponding **_number_**. We know that this **_number_** corresponds to the return obtained by using the policy corresponding to θ\\thetaθ to collect an episode, but the algorithms are not aware of this. To the algorithms, the way we evaluate θ\\thetaθ is considered a black box, and they don't worry about the details. The algorithms only care about finding the value of θ\\thetaθ that will maximize the number that comes out of the black box.

In the video below, you'll learn about a couple more black-box optimization techniques, to include the **cross-entropy method** and **[evolution strategies](https://blog.openai.com/evolution-strategies/)**.



# OpenAI Request for Research

So far in this lesson, you have learned about many black-box optimization techniques for finding the optimal policy. Run each algorithm for many random seeds, to test stability.

Take the time now to implement some of them, and compare performance on OpenAI Gym's `CartPole-v0` environment.

> **Note**: This suggested exercise is completely optional.

Once you have completed your analysis, you're encouraged to write up your own blog post that responds to [OpenAI's Request for Research](https://openai.com/requests-for-research/#cartpole)! (_This request references policy gradient methods. You'll learn about policy gradient methods in the next lesson._)

![OpenAI Gym's CartPole environment](https://video.udacity-data.com/topher/2018/June/5b3790c2_cartpole/cartpole.gif)

OpenAI Gym's CartPole environment

Implement (vanilla) hill climbing and steepest ascent hill climbing, both with simulated annealing and adaptive noise scaling.

If you also want to compare the performance to evolution strategies, you can find a well-written implementation [here](https://github.com/alirezamika/evostra). To see how to apply it to an OpenAI Gym task, check out [this repository](https://github.com/alirezamika/bipedal-es).

To see one way to structure your analysis, check out [this blog post](http://kvfrans.com/simple-algoritms-for-solving-cartpole/), along with the [accompanying code](https://github.com/kvfrans/openai-cartpole).

For instance, you will likely find that hill climbing is very unstable, where the number of episodes that it takes to solve `CartPole-v0` varies greatly with the random seed. (_Check out the figure below!_)

![Histogram of number of episodes needed to solve CartPole with hill climbing.](https://video.udacity-data.com/topher/2018/June/5b357385_screen-shot-2018-06-28-at-6.46.54-pm/screen-shot-2018-06-28-at-6.46.54-pm.png)

Histogram of number of episodes needed to solve CartPole with hill climbing. ([Source](http://kvfrans.com/simple-algoritms-for-solving-cartpole/))



# Summary

![Objective function](https://video.udacity-data.com/topher/2018/June/5b3270ab_screen-shot-2018-06-26-at-11.53.35-am/screen-shot-2018-06-26-at-11.53.35-am.png)

Objective function

### Policy-Based Methods

* * *

- With **value-based methods**, the agent uses its experience with the environment to maintain an estimate of the optimal action-value function. The optimal policy is then obtained from the optimal action-value function estimate.
- **Policy-based methods** directly learn the optimal policy, without having to maintain a separate value function estimate.

### Policy Function Approximation

* * *

- In deep reinforcement learning, it is common to represent the policy with a neural network.
    - This network takes the environment state as **_input_**.
    - If the environment has discrete actions, the **_output_** layer has a node for each possible action and contains the probability that the agent should select each possible action.
- The weights in this neural network are initially set to random values. Then, the agent updates the weights as it interacts with (_and learns more about_) the environment.

### More on the Policy

* * *

- Policy-based methods can learn either stochastic or deterministic policies, and they can be used to solve environments with either finite or continuous action spaces.

### Hill Climbing

* * *

- **Hill climbing** is an iterative algorithm that can be used to find the weights θ\\thetaθ for an optimal policy.
- At each iteration,
    - We slightly perturb the values of the current best estimate for the weights θbest\\theta\_{best}θbest​, to yield a new set of weights.
    - These new weights are then used to collect an episode. If the new weights θnew\\theta\_{new}θnew​ resulted in higher return than the old weights, then we set θbest←θnew\\theta\_{best} \\leftarrow \\theta\_{new}θbest​←θnew​.

### Beyond Hill Climbing

* * *

- **Steepest ascent hill climbing** is a variation of hill climbing that chooses a small number of neighboring policies at each iteration and chooses the best among them.
- **Simulated annealing** uses a pre-defined schedule to control how the policy space is explored, and gradually reduces the search radius as we get closer to the optimal solution.
- **Adaptive noise scaling** decreases the search radius with each iteration when a new best policy is found, and otherwise increases the search radius.

### More Black-Box Optimization

* * *

- The **cross-entropy method** iteratively suggests a small number of neighboring policies, and uses a small percentage of the best performing policies to calculate a new estimate.
- The **evolution strategies** technique considers the return corresponding to each candidate policy. The policy estimate at the next iteration is a weighted sum of all of the candidate policies, where policies that got higher return are given higher weight.

### Why Policy-Based Methods?

* * *

- There are three reasons why we consider policy-based methods:
    1. **Simplicity**: Policy-based methods directly get to the problem at hand (estimating the optimal policy), without having to store a bunch of additional data (i.e., the action values) that may not be useful.
    2. **Stochastic policies**: Unlike value-based methods, policy-based methods can learn true stochastic policies.
    3. **Continuous action spaces**: Policy-based methods are well-suited for continuous action spaces.

