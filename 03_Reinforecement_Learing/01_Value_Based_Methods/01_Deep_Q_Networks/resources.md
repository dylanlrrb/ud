# From RL to Deep RL

In the video below, **Kelvin Lwin** will introduce you to the idea of using neural networks to expand the size of the problems that we can solve with reinforcement learning. This context is useful preparation for exploring the details behind the Deep Q-Learning algorithm later in this lesson!

_Kelvin is a Senior Deep Learning Instructor at the_ [_NVIDIA Deep Learning Institute_](https://www.nvidia.com/en-us/deep-learning-ai/education)_._

## Stabilizing Deep Reinforcement Learning

* * *

As you'll learn in this lesson, the Deep Q-Learning algorithm represents the optimal action-value function q∗q\_\*q∗​ as a neural network (instead of a table).

Unfortunately, reinforcement learning is [notoriously unstable](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.73.3097&rep=rep1&type=pdf) when neural networks are used to represent the action values. In this lesson, you'll learn all about the Deep Q-Learning algorithm, which addressed these instabilities by using **two key features**:

- Experience Replay
- Fixed Q-Targets

Watch the video below to learn more!

## Additional References

* * *

- Riedmiller, Martin. "Neural fitted Q iteration–first experiences with a data efficient neural reinforcement learning method." European Conference on Machine Learning. Springer, Berlin, Heidelberg, 2005. [http://ml.informatik.uni-freiburg.de/former/\_media/publications/rieecml05.pdf](http://ml.informatik.uni-freiburg.de/former/_media/publications/rieecml05.pdf)
    
- Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature518.7540 (2015): 529. [http://www.davidqiu.com:8888/research/nature14236.pdf](http://www.davidqiu.com:8888/research/nature14236.pdf)


# Lesson Outline

In this lesson, we will cover the following topics:

- How DQN works
- Experience replay
- Fixed-Q targets
- Deep Q-learning algorithm and how to implement it
- Deep Q-learning improvements
    - Double DQN
    - Prioritized experience replay
    - Dueling DQN
    - Rainbow


## Summary

* * *

When the agent interacts with the environment, the sequence of experience tuples can be highly correlated. The naive Q-learning algorithm that learns from each of these experience tuples in sequential order runs the risk of getting swayed by the effects of this correlation. By instead keeping track of a **replay buffer** and using **experience replay** to sample from the buffer at random, we can prevent action values from oscillating or diverging catastrophically.

The **replay buffer** contains a collection of experience tuples (SSS, AAA, RRR, S′S'S′). The tuples are gradually added to the buffer as we are interacting with the environment.

The act of sampling a small batch of tuples from the replay buffer in order to learn is known as **experience replay**. In addition to breaking harmful correlations, experience replay allows us to learn more from individual tuples multiple times, recall rare occurrences, and in general make better use of our experience.


## Summary

* * *

In Q-Learning, we **_update a guess with a guess_**, and this can potentially lead to harmful correlations. To avoid this, we can update the parameters www in the network q^\\hat{q}q^​ to better approximate the action value corresponding to state SSS and action AAA with the following update rule:

Δw\=α⋅(R+γmaxaq^(S′,a,w−)⎵TD target−q^(S,A,w)⎵old value)⏞TD error∇wq^(S,A,w)\\Delta w = \\alpha \\cdot \\overbrace{( \\underbrace{R + \\gamma \\max\_a\\hat{q}(S', a, w^-)}\_{\\rm {TD~target}} - \\underbrace{\\hat{q}(S, A, w)}\_{\\rm {old~value}})}^{\\rm {TD~error}} \\nabla\_w\\hat{q}(S, A, w)Δw\=α⋅(TD target R+γamax​q^​(S′,a,w−)​​−old value q^​(S,A,w)​​) ​TD error​∇w​q^​(S,A,w)

where w−w^-w− are the weights of a separate target network that are not changed during the learning step.

And (SSS, AAA, RRR, S′S'S′) is an experience tuple.

**Note**: Ever wondered how the example in the video would look in real life? See: [Carrot Stick Riding](https://www.youtube.com/watch?v=-PVFBGN_zoM).


# Deep Q-Learning Algorithm

Please take the time now to read the [research paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) that introduces the Deep Q-Learning algorithm.

## Reading Scientific Papers

* * *

As part of this nanodegree, you will learn about many of the most recent, cutting-edge algorithms! Because of this, it will prove useful to learn how to read the original research papers. Here are some [excellent tips](https://www.huffingtonpost.com/jennifer-raff/how-to-read-and-understand-a-scientific-paper_b_5501628.html). Some of the best advice is:

- Take notes.
- Read the paper multiple times. On the first couple readings, try to focus on the main points:
    1. What kind of tasks are the authors using deep reinforcement learning (RL) to solve? What are the states, actions, and rewards?
    2. What neural network architecture is used to approximate the action-value function?
    3. How are experience replay and fixed Q-targets used to stabilize the learning algorithm?
    4. What are the results?
- Understanding the paper will probably take you longer than you think. Be patient, and reach out to the Udacity community with any questions.


# Deep Q-Learning Improvements

Several improvements to the original Deep Q-Learning algorithm have been suggested. Over the next several videos, we'll look at three of the more prominent ones.

## Double DQN

* * *

Deep Q-Learning [tends to overestimate](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf) action values. [Double Q-Learning](https://arxiv.org/abs/1509.06461) has been shown to work well in practice to help with this.

## Prioritized Experience Replay

* * *

Deep Q-Learning samples experience transitions _uniformly_ from a replay memory. [Prioritized experienced replay](https://arxiv.org/abs/1511.05952) is based on the idea that the agent can learn more effectively from some transitions than from others, and the more important transitions should be sampled with higher probability.

## Dueling DQN

* * *

Currently, in order to determine which states are (or are not) valuable, we have to estimate the corresponding action values _for each action_. However, by replacing the traditional Deep Q-Network (DQN) architecture with a [dueling architecture](https://arxiv.org/abs/1511.06581), we can assess the value of each state, without having to learn the effect of each action.


## Overestimation of Q-values

The first problem we are going to address is the overestimation of action values that Q-learning is prone to.

The update rule for Q-learning with function approximation is

Δw\=α(R+γmaxaq^(S′,a,w)−q^(S,A,w))∇wq^(S,A,w)\\Delta {w} = \\alpha(R+\\gamma\\max\_{a}\\hat{q}(S^{\\prime},a,{w})-\\hat{q}(S,{A},{w}))\\nabla\_{w}\\hat{q}(S,{A},{w})Δw\=α(R+γamax​q^​(S′,a,w)−q^​(S,A,w))∇w​q^​(S,A,w)

where R+γmaxaq^(S′,a,w)R+\\gamma\\max\_{a}\\hat{q}(S^{\\prime},a,{w})R+γmaxa​q^​(S′,a,w) is the TD target.

  

## TD Target

To better under the maxmaxmax operation in TD target, we write the formula for TD target and expand the max operation

R+γq^(S′,argmaxaq^(S′,a,w),w)R+{\\gamma}{\\hat{q}}(S^{\\prime},\\arg\\max\_{a}\\hat{q}(S^{\\prime},a,{w}),w)R+γq^​(S′,argamax​q^​(S′,a,w),w)

It's possible for the argmax\\arg\\maxargmax operation to make mistake, especially in the early stages. This is because the Q-value q^\\hat{q}q^​ is still evolving, we may not have gathered enough information to figure out the best action. The accuracy of Q-values depends a lot on what what actions have been tried, and what neighboring states have been explored.

  

## Double Q-Learning

Double Q-learning can make estimation more robust by selecting the best action using one set of parameters www, but evaluating it using a different set of parameters w′w^{\\prime}w′.

R+γq^(S′,argmaxaq^(S′,a,w),w′)R+{\\gamma}{\\hat{q}}(S^{\\prime},\\arg\\max\_{a}\\hat{q}(S^{\\prime},a,{w}),w^{\\prime})R+γq^​(S′,argamax​q^​(S′,a,w),w′)

Where do we get second set of parameters w′w^{\\prime}w′ from?

- In the original formula of double Q-learning, two value functions are basically maintained, and randomly choose one of them to update at each step using the other only for evaluating actions.
- When using DQNs with fixed Q targets, we already have an alternate set of parameters w−w^-w−. Since w−w^-w− has been kept frozen for a while, it is different enough from www that it can be reused for this purpose.

## Notes

* * *

You can read more about Double DQN (DDQN) by perusing this [research paper](https://arxiv.org/abs/1509.06461).

If you'd like to dig deeper into how Deep Q-Learning overestimates action values, please read this [research paper](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf).


# Prioritized Experience Replay

**TD error delta**

- Criteria used to assign priorities to each tuple
- Formula: δt\=Rt+1+γmaxaq^(St+1,a,w)−q^(St,At,w)\\delta\_t = R\_{t+1} + \\gamma{max\_a{\\hat{q}(S\_{t+1},a,w)}-{\\hat{q}(S\_{t},A\_t,w)}} δt​\=Rt+1​+γmaxa​q^​(St+1​,a,w)−q^​(St​,At​,w)
- The bigger the error, the more we expect to learn from that tuple

**Measure of Priority**

- The magnitude of TD error
- Formula: pt\=∣δt∣p\_t = |\\delta\_t|pt​\=∣δt​∣
- Priority is stored along with each corresponding tuple in the replay buffer

**Sampling Probability**

- Computed from priority when creating batches
- Formula: P(i)\=pi∑kpkP(i)=\\frac{p\_i}{\\sum\_k{p\_k}}P(i)\=∑k​pk​pi​​

# Improvement on Prioritized Experience Relay

**TD Error is Zero**

- **Problem**: If the TD error is zero, then the priority value of the tuple and hence its probability of being picked will also be zero. This doesn't necessarily mean we have nothing more to learn from such a tuple. It might be the case that our estimate was closed due to the limited samples we visited till that point.
- **Solution**: To prevent tuples from being starved for selection, we can add a small constant eee to every priority value. Thus, **priority** will be expressed as

pt\=∣δt∣+ep\_t=|\\delta\_t|+ept​\=∣δt​∣+e

**Greedy Usage of Priority Values**

- **Problem**: Greedily using priority values may lead to a small subset of experiences being relayed over and over, resulting in a overfitting to that subset.
- **Solution**: Reintroduce some element of uniform random sampling. This adds another hyperparameter aaa which we use to redefine the sample probability as

P(i)\=pia∑kpkaP(i)=\\frac{{p\_i}^a}{\\sum\_k{p\_k}^a}P(i)\=∑k​pk​api​a​

# Adjustment to the Update Rule

We we use prioritized experience relay, we have to make one adjustment to our update rule, which is

Δw\=α(1N⋅1Pi)bδi∇wq^(Si,Ai,w)\\Delta{w}=\\alpha(\\frac{1}{N}\\cdot\\frac{1}{P{i}})^b{\\delta\_i}\\nabla\_w\\hat{q}(S\_i,A\_i,w)Δw\=α(N1​⋅Pi1​)bδi​∇w​q^​(Si​,Ai​,w)

where (1N⋅1Pi)b(\\frac{1}{N}\\cdot\\frac{1}{P{i}})^b(N1​⋅Pi1​)b stands for the importance-sampling weight.

  

## Notes

* * *

You can read more about prioritized experience replay by perusing this [research paper](https://arxiv.org/abs/1511.05952).


# Dueling DQN

The core idea of dueling networks is to use two streams

- one stream estimates the **state value function**: V(s)V(s)V(s)
- one stream estimates the **advantage for each action**: A(s,a)A(s,a)A(s,a)

Finally, by combining the state and advantage values, we are able to obtain the desired **Q-values**:

Q(s,a)\=V(s)+A(s,a)Q(s,a) = V(s)+A(s,a)Q(s,a)\=V(s)+A(s,a)

  

## Notes

* * *

You can read more about Dueling DQN by perusing this [research paper](https://arxiv.org/abs/1511.06581).


# Rainbow

So far, you've learned about three extensions to the Deep Q-Networks (DQN) algorithm:

- Double DQN (DDQN)
- Prioritized experience replay
- Dueling DQN

But these aren't the only extensions to the DQN algorithm! Many more extensions have been proposed, including:

- Learning from [multi-step bootstrap targets](https://arxiv.org/abs/1602.01783)
- [Distributional DQN](https://arxiv.org/abs/1707.06887)
- [Noisy DQN](https://arxiv.org/abs/1706.10295)

Each of the six extensions address a **_different_** issue with the original DQN algorithm.

Researchers at Google DeepMind recently tested the performance of an agent that incorporated all six of these modifications. The corresponding algorithm was termed [Rainbow](https://arxiv.org/abs/1710.02298).

It outperforms each of the individual modifications and achieves state-of-the-art performance on Atari 2600 games!

![Performance on Atari games: comparison of Rainbow to six baselines.](https://video.udacity-data.com/topher/2018/June/5b3814f1_screen-shot-2018-06-30-at-6.40.09-pm/screen-shot-2018-06-30-at-6.40.09-pm.png)

Performance on Atari games: comparison of Rainbow to six baselines.

## In Practice

* * *

In mid-2018, OpenAI held [a contest](https://contest.openai.com/), where participants were tasked to create an algorithm that could learn to play the [Sonic the Hedgehog](https://en.wikipedia.org/wiki/Sonic_the_Hedgehog) game. The participants were tasked to train their RL algorithms on provided game levels; then, the trained agents were ranked according to their performance on previously unseen levels.

Thus, the contest was designed to assess the ability of trained RL agents to generalize to new tasks.

![Sonic The Hedgehog ([Source](https://contest.openai.com))](https://video.udacity-data.com/topher/2018/June/5b381932_sonic/sonic.gif)

Sonic The Hedgehog ([Source](https://contest.openai.com/))

One of the provided baseline algorithms was **Rainbow DQN**. If you'd like to play with this dataset and run the baseline algorithms, you're encouraged to follow the [setup instructions](https://contest.openai.com/2018-1/details/).

![Baseline results on the Retro Contest (test set)([Source](https://blog.openai.com/retro-contest/))](https://video.udacity-data.com/topher/2018/July/5b381a72_screen-shot-2018-06-30-at-7.03.40-pm/screen-shot-2018-06-30-at-7.03.40-pm.png)

Baseline results on the Retro Contest (test set) ([Source](https://blog.openai.com/retro-contest/))
