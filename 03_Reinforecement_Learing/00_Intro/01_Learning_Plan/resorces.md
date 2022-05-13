# Learning Plan

The first part of this Nanodegree program covers the basics of reinforcement learning and lasts **4 weeks**. For these first 4 weeks of the program, you will build a strong background in reinforcement learning (_without neural networks_), before spending the remaining time in the course learning how to leverage neural networks to train intelligent agents.

## Week 1

* * *

In the first week, you will learn the basics of reinforcement learning.

#### Lesson: Introduction to RL

In this lesson, you'll explore a friendly introduction to reinforcement learning.

#### Lesson: The RL Framework: The Problem

In this lesson, you'll learn how to specify a real-world problem as a Markov Decision Process (MDP), so that it can be solved with reinforcement learning.

#### Lesson: The RL Framework: The Solution

In this lesson, you'll learn all about value functions and optimal policies.

#### Readings

- **Chapter 1** (especially 1.1-1.4) of the [textbook](http://go.udacity.com/rl-textbook)
- **Chapter 3** (especially 3.1-3.3, 3.5-3.6) of the [textbook](http://go.udacity.com/rl-textbook)

## Week 2

* * *

In the second week, you'll build your own agents to solve the reinforcement learning problem.

#### Lesson: Dynamic Programming (_Optional_)

In this lesson, you'll build some intuition for the reinforcement learning problem by learning about a class of solution methods that solve a slightly easier problem. (_This lesson is **optional** and can be accessed in the **extracurricular content**._)

#### Lesson: Monte Carlo Methods

In this lesson, you'll learn about a class of solution methods known as Monte Carlo methods. You'll implement your own [Blackjack](https://en.wikipedia.org/wiki/Blackjack)\-playing agent in OpenAI Gym.

#### Readings

- **Chapter 4** (especially 4.1-4.4) of the [textbook](http://go.udacity.com/rl-textbook) (_This reading is **optional** and accompanies the optional **Dynamic Programming** lesson._)
- **Chapter 5** (especially 5.1-5.6) of the [textbook](http://go.udacity.com/rl-textbook)

## Week 3

* * *

In the third week, you'll leverage a slightly more sophisticated class of solution methods to build your own agents in OpenAI Gym.

#### Lesson: Temporal-Difference Methods

In this lesson, you'll learn how to apply temporal-difference methods such as SARSA, Q-learning, and Expected SARSA to solve both episodic and continuing tasks.

#### Lesson: Solve OpenAI Gym's Taxi-v2 Task

In this lesson, you'll apply what you've learned to train a taxi to pick up and drop off passengers.

#### Readings

- **Chapter 6** (especially 6.1-6.6) of the [textbook](http://go.udacity.com/rl-textbook)
- **Subsection 3.1** of this [research paper](https://arxiv.org/abs/cs/9905014)

## Week 4

* * *

In the last week, you will learn how to adapt the algorithms that you've been learning about to solve a larger class of problems.

#### Lesson: RL in Continuous Spaces

In this lesson, you'll explore how to use techniques such as tile coding and coarse coding to expand the size of the problems that can be solved with traditional reinforcement learning algorithms.

#### Lesson: What's Next?

In this lesson, you'll learn more about what's to come in the next three parts of the Nanodegree program. You'll also get some tips for how to best spend your time, if you finish this first part of the Nanodegree program early!

#### Readings

- **Chapter 9** (especially 9.1-9.7) of the [textbook](http://go.udacity.com/rl-textbook)

![](https://video.udacity-data.com/topher/2020/October/5f862704_screen-shot-2020-10-13-at-3.15.21-pm/screen-shot-2020-10-13-at-3.15.21-pm.png)


# Reference Guide

You are encouraged to download [this sheet](https://github.com/udacity/deep-reinforcement-learning/blob/master/cheatsheet/cheatsheet.pdf), which contains most of the notation and algorithms that we will use in the first part of this course. Please only use this sheet as a supplement to your own notes! :)

Another useful notation guide can be found in the pages immediately preceding Chapter 1 of the [textbook](http://go.udacity.com/rl-textbook).


# OpenAI Gym

In this course, we’ll make extensive use of [OpenAI Gym](https://gym.openai.com/), an open-source toolkit created by [OpenAI](https://openai.com/) for developing and comparing reinforcement learning (RL) algorithms. Using this toolkit, you will teach your own agents to accomplish a variety of complex tasks.

We'll use it to build practical, hands-on skills with many of the algorithms that we'll learn about during the nanodegree. Then, once you build your confidence with these environments, you'll be more than ready to tackle the projects!

## Take a Look

* * *

You can take a look at some of the environments that OpenAI Gym offers from the link [here](https://gym.openai.com/envs/).

![](https://video.udacity-data.com/topher/2018/April/5ae0c33a_openaigym-gif/openaigym-gif.gif)

If you'd like to learn more about OpenAI Gym, please peruse the [GitHub repository](https://github.com/openai/gym.git). Check out this [blog post](https://blog.openai.com/openai-gym-beta/) to read more about how OpenAI Gym is used to accelerate reinforcement learning (RL) research.

## Learning by Doing

* * *

In the first part of the nanodegree, we’ll be using many of the environments that are available as part of this toolkit:

- You will deepen your understanding of **Dynamic Programming** with the frozen lake environment. As part of this environment, you'll teach an agent to navigate a world without falling into pits of frozen water.
- When learning about **Monte Carlo** methods, you'll write an agent to play blackjack.
- As part of the lesson on **Temporal-Difference** methods, you will teach an agent to navigate a world with a large cliff, where your goal is to avoid falling in!
- In the **Solve OpenAI Gym's Taxi-v2 Task** lesson, you will train a taxi to pick up and drop off passengers as quickly as possible.
- In **RL in Continuous Spaces**, you will train an under-powered car to manage a large hill.

One of the really cool things about OpenAI Gym is that you can watch your performance. So, your agent might start off just behaving randomly, but as it learns from interaction, you’ll be able to see it choose actions in a much more calculated, intelligent way.

![](https://video.udacity-data.com/topher/2018/April/5ae40b5a_image4/image4.gif)

A well-trained OpenAI Gym agent ([Source](https://blog.openai.com/roboschool/))

What’s also really cool is that if you’re happy with how smart you’ve made your agents, or how quickly you’ve made your agents learn, you can upload your implementations to share knowledge with the world! Check out the [leaderboard](https://github.com/openai/gym/wiki/Leaderboard), which contains the best solutions to each task.

## Installation Instructions

* * *

You are **not** required to install OpenAI Gym on your computer, and you will have the option to do all of your coding implementations within the classroom.

If you'd like to install OpenAI Gym on your machine, you are encouraged to follow the instructions in the [GitHub repository](https://github.com/openai/gym#installation).


# GitHub Repository

You can find all of the coding exercises from the lessons in this [GitHub repository](https://github.com/udacity/deep-reinforcement-learning).

To clone the repository:

```
git clone https://github.com/udacity/deep-reinforcement-learning.git
```

## Hands-on Learning

* * *

As a student in this program, you'll train your own agents to accomplish a variety of complex tasks!

![](https://video.udacity-data.com/topher/2018/July/5b44d455_42135602-b0335606-7d12-11e8-8689-dd1cf9fa11a9/42135602-b0335606-7d12-11e8-8689-dd1cf9fa11a9.gif)

# Student Resources

As you learn more about deep reinforcement learning and progress through this course, you'll likely find a number of outside resources that are useful for supplementing your study.

You can start with the student-curated list of resources at [this link](http://bit.ly/drlndlinks). Please contribute to the list, if you find more to add!

## Books to Read

* * *

We believe that you learn best when you are exposed to multiple perspectives on the same idea. As such, we recommend checking out the books below to get an added perspective on Deep Reinforcement Learning.

- [**Grokking Deep Reinforcement Learning**](https://www.manning.com/books/grokking-deep-reinforcement-learning) by Miguel Morales. Use our exclusive discount code **gdrludacity50** for 50% off. This textbook features numerous examples, illustrations, exercises, and crystal-clear teaching. It is currently under development, but you can sign up to read the chapters as they are written!
    
- [**Reinforcement Learning: An Introduction**](http://go.udacity.com/rl-textbook) by Richard S. Sutton and Andrew G. Barto. This book is a classic text with an excellent introduction to reinforcement learning fundamentals.

