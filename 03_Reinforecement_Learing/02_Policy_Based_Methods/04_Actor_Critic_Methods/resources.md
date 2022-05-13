# Important links

- I partnered with Udacity and Manning to give all students of this Nanodegree the opportunity to obtain a copy of my book at a 40% discount. The more perspectives you get on deep reinforcement learning, the better for you. If you'd like to give it a try, go get chapter 01 for free [here](http://bit.ly/gdrl_u). Make sure to use the discount code: 'gdrludacity40' at checkout for a 40% off.
- I'm very proud to work for Lockheed Martin. Working for a company that does work that matters with a huge impact on the lives of many people is very rewarding (no pun intended.) If you are interested in joining us, check out this [link](http://bit.ly/lmco_u). We are always looking for talented individuals with experience in AI (and deep reinforcement learning).
- Last, but not least. I'm an Instructional Associate at Georgia Tech for the Reinforcement Learning and Decision Making graduate course available on the OMSCS program that Udacity helped launch. If you'd like more information regarding this program, take a look [here](http://bit.ly/omscs_u).


# Baselines and Critics

The argument you often hear as to why to call a neural network trained with Monte-Carlo estimates a "Critic" is because function approximators, such as a neural network, are biased as a byproduct that they are not perfect. That's a fair point, though, I prefer the distinction based on whether we pick a Monte-Carlo or a TD estimate to train our function approximator. Now, definitely we should not be calling Actor-Critic methods every method that uses 2 neural networks. You'll be surprised!

The important takeaway for you, though, is that there are inconsistencies out there. You often see methods named "Actor-Critic" when they are not. I just want to bring the issue to your attention.


# A Basic Actor-Critic Agent

One important thing to note here is that I use V(s;θv)V(s;\\theta\_v)V(s;θv​) or A(s,a)A(s,a)A(s,a), but sometimes Vπ(s;θv)V\_\\pi(s;\\theta\_v)Vπ​(s;θv​) or Aπ(s,a)A\_\\pi(s,a)Aπ​(s,a) (see the π\\piπ there? See the θv\\theta\_vθv​? What's going on?)

There are 2 thing actually going on in there.

1. A very common thing you'll see in reinforcement learning is the oversimplification of notation. However, both styles, whether you see A(s,a)A(s,a)A(s,a), or Aπ(s,a)A\_\\pi(s,a)Aπ​(s,a) (value functions with or without a π\\piπ,) it means you are evaluating a value function of policy π\\piπ. In the case of AAA, the advantage function. A different case would be when you see a superscript ∗\*∗. For example, A∗(s,a)A^\*(s,a)A∗(s,a) means the optimal advantage function. Q-learning learns the optimal action-value function, Q∗(s,a)Q^\*(s,a)Q∗(s,a), for example.
2. The other thing is the use of θv\\theta\_vθv​ in some value functions and not in others. This only means that such value function is using a neural network. For example, V(s;θv)V(s;\\theta\_v)V(s;θv​) is using a neural network as a function approximator, but A(s,a)A(s,a)A(s,a) is not. We are calculating the advantage function A(s,a)A(s, a)A(s,a) using the state-value function V(s;θv)V(s;\\theta\_v)V(s;θv​), but A(s,a)A(s, a)A(s,a) is not using function approximation directly.


Link to the Q-Prop paper: [https://arxiv.org/abs/1611.02247](https://arxiv.org/abs/1611.02247)


Link to the GAE paper: [https://arxiv.org/abs/1506.02438](https://arxiv.org/abs/1506.02438)


# DDPG: Deep Deterministic Policy Gradient, Continuous Action-space

In the [DDPG paper](https://arxiv.org/abs/1509.02971), they introduced this algorithm as an "Actor-Critic" method. Though, some researchers think DDPG is best classified as a DQN method for continuous action spaces (along with [NAF](https://arxiv.org/abs/1603.00748)). Regardless, DDPG is a very successful method and it's good for you to gain some intuition.


# Summary

# Final Reminders

- Make sure you take advantage of the discount Udacity is able to bring you on [Grokking Deep Reinforcement Learning](http://bit.ly/gdrl_u): remember, 'gdrludacity40' gives you a 40% off.
- Finally, life is better together. Make sure to connect [here](http://bit.ly/mimoralea_t), [here](http://bit.ly/mimoralea_l), or [some other way](http://bit.ly/mimoralea).

Good luck to you going forward!

