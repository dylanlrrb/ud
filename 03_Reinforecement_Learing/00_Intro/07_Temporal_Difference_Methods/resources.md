This lesson covers material in **Chapter 6** (especially 6.1-6.6) of the [textbook](http://go.udacity.com/rl-textbook).


# Review: MC Control Methods

In the previous lesson, you learned about the **control problem** in reinforcement learning and implemented some Monte Carlo (MC) control methods.

> **Control Problem**: Estimate the optimal policy.

In this lesson, you will learn several techniques for Temporal-Difference (TD) control.

## Review

* * *

Before continuing, please review **Constant-alpha MC Control** from the previous lesson.

Remember that the constant-α\\alphaα MC control algorithm alternates between **policy evaluation** and **policy improvement** steps to recover the optimal policy π∗\\pi\_\*π∗​.

![Constant-alpha MC Control](https://video.udacity-data.com/topher/2018/May/5af4d181_screen-shot-2018-05-10-at-6.10.16-pm/screen-shot-2018-05-10-at-6.10.16-pm.png)

Constant-alpha MC Control

In the **policy evaluation** step, the agent collects an episode S0,A0,R1,…,STS\_0, A\_0, R\_1, \\ldots, S\_TS0​,A0​,R1​,…,ST​ using the most recent policy π\\piπ. After the episode finishes, for each time-step ttt, if the corresponding state-action pair (St,At)(S\_t,A\_t)(St​,At​) is a first visit, the Q-table is modified using the following **update equation**:

Q(St,At)←Q(St,At)+α(Gt−Q(St,At))Q(S\_t,A\_t) \\leftarrow Q(S\_t,A\_t) + \\alpha(G\_t - Q(S\_t, A\_t))Q(St​,At​)←Q(St​,At​)+α(Gt​−Q(St​,At​))

where Gt:\=∑s\=t+1Tγs−t−1RsG\_t := \\sum\_{s={t+1}}^T\\gamma^{s-t-1}R\_sGt​:\=∑s\=t+1T​γs−t−1Rs​ is the return at timestep ttt, and Q(St,At)Q(S\_t,A\_t)Q(St​,At​) is the entry in the Q-table corresponding to state StS\_tSt​ and action AtA\_tAt​.

The main idea behind this **update equation** is that Q(St,At)Q(S\_t,A\_t)Q(St​,At​) contains the agent's estimate for the expected return if the environment is in state StS\_tSt​ and the agent selects action AtA\_tAt​. If the return GtG\_tGt​ is **not** equal to Q(St,At)Q(S\_t,A\_t)Q(St​,At​), then we push the value of Q(St,At)Q(S\_t,A\_t)Q(St​,At​) to make it agree slightly more with the return. The magnitude of the change that we make to Q(St,At)Q(S\_t,A\_t)Q(St​,At​) is controlled by the hyperparameter α\>0\\alpha>0α\>0.


# TD Control: Sarsa

Monte Carlo (MC) control methods require us to complete an entire episode of interaction before updating the Q-table. Temporal Difference (TD) methods will instead update the Q-table after every time step.

## Video

* * *

Watch the next video to learn about **Sarsa** (or **Sarsa(0)**), one method for TD control.

## Video

* * *

## Pseudocode

* * *

![](https://video.udacity-data.com/topher/2018/May/5aece99b_screen-shot-2018-05-04-at-6.14.28-pm/screen-shot-2018-05-04-at-6.14.28-pm.png)

In the algorithm, the number of episodes the agent collects is equal to num\_episodesnum\\\_episodesnum\_episodes. For every time step t≥0t\\geq 0t≥0, the agent:

- **takes** the action AtA\_tAt​ (from the current state StS\_tSt​) that is ϵ\\epsilonϵ\-greedy with respect to the Q-table,
- receives the reward Rt+1R\_{t+1}Rt+1​ and next state St+1S\_{t+1}St+1​,
- **chooses** the next action At+1A\_{t+1}At+1​ (from the next state St+1S\_{t+1}St+1​) that is ϵ\\epsilonϵ\-greedy with respect to the Q-table,
- uses the information in the tuple (StS\_tSt​, AtA\_tAt​, Rt+1R\_{t+1}Rt+1​, St+1S\_{t+1}St+1​, At+1A\_{t+1}At+1​) to update the entry Q(St,At)Q(S\_t, A\_t)Q(St​,At​) in the Q-table corresponding to the current state StS\_tSt​ and the action AtA\_tAt​.


# TD Control: Q-Learning

Please watch the video below to learn about **Q-Learning (or Sarsamax)**, a second method for TD control.

Check out this (optional) [research paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.80.7501&rep=rep1&type=pdf) to read the proof that Q-Learning (or Sarsamax) converges.

## Pseudocode

* * *

![](https://video.udacity-data.com/topher/2018/May/5aece9b8_screen-shot-2018-05-04-at-6.14.42-pm/screen-shot-2018-05-04-at-6.14.42-pm.png)


# TD Control: Expected Sarsa

Please watch the video below to learn about **Expected Sarsa**, a third method for TD control.

Check out this (optional) [research paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.216.4144&rep=rep1&type=pdf) to learn more about Expected Sarsa.

## Pseudocode

* * *

![](https://video.udacity-data.com/topher/2018/May/5aece9d8_screen-shot-2018-05-04-at-6.14.56-pm/screen-shot-2018-05-04-at-6.14.56-pm.png)


![Exploration-Exploitation Dilemma](https://video.udacity-data.com/topher/2017/October/59d55ce3_exploration-vs.-exploitation/exploration-vs.-exploitation.png)

Exploration-Exploitation Dilemma ([Source](http://slides.com/ericmoura/deck-2/embed))

# TD Control: Theory and Practice

## Greedy in the Limit with Infinite Exploration (GLIE)

* * *

The **Greedy in the Limit with Infinite Exploration (GLIE)** conditions were introduced in the previous lesson, when we learned about MC control. There are many ways to satisfy the GLIE conditions, all of which involve gradually decaying the value of ϵ\\epsilonϵ when constructing ϵ\\epsilonϵ\-greedy policies.

In particular, let ϵi\\epsilon\_iϵi​ correspond to the iii\-th time step. Then, to satisfy the GLIE conditions, we need only set ϵi\\epsilon\_iϵi​ such that:

- ϵi\>0\\epsilon\_i > 0ϵi​\>0 for all time steps iii, and
- ϵi\\epsilon\_iϵi​ decays to zero in the limit as the time step iii approaches infinity (that is, limi→∞ϵi\=0\\lim\_{i\\to\\infty} \\epsilon\_i = 0limi→∞​ϵi​\=0),

## In Theory

* * *

All of the TD control algorithms we have examined (Sarsa, Sarsamax, Expected Sarsa) are **guaranteed to converge** to the optimal action-value function q∗q\_\*q∗​, as long as the step-size parameter α\\alphaα is sufficiently small, and the GLIE conditions are met.

Once we have a good estimate for q∗q\_\*q∗​, a corresponding optimal policy π∗\\pi\_\*π∗​ can then be quickly obtained by setting π∗(s)\=argmaxa∈A(s)q∗(s,a)\\pi\_\*(s) = \\arg\\max\_{a\\in\\mathcal{A}(s)} q\_\*(s, a)π∗​(s)\=argmaxa∈A(s)​q∗​(s,a) for all s∈Ss\\in\\mathcal{S}s∈S.

## In Practice

* * *

In practice, it is common to completely ignore the GLIE conditions and still recover an optimal policy. (_You will see an example of this in the solution notebook._)

## Optimism

* * *

You have learned that for any TD control method, you must begin by initializing the values in the Q-table. It has been shown that [initializing the estimates to large values](http://papers.nips.cc/paper/1944-convergence-of-optimistic-and-incremental-q-learning.pdf) can improve performance. For instance, if all of the possible rewards that can be received by the agent are negative, then initializing every estimate in the Q-table to zeros is a good technique. In this case, we refer to the initialized Q-table as **optimistic**, since the action-value estimates are guaranteed to be larger than the true action values.


# OpenAI Gym: CliffWalkingEnv

In order to master the algorithms discussed in this lesson, you will write your own implementations in Python. While your code will be designed to work with any OpenAI Gym environment, you will test your code with the CliffWalking environment.

![](https://video.udacity-data.com/topher/2017/October/59de4705_matengai-of-kuniga-coast-in-oki-island-shimane-pref600/matengai-of-kuniga-coast-in-oki-island-shimane-pref600.jpg)

Source: Wikipedia

In the CliffWalking environment, the agent navigates a 4x12 gridworld. Please read about the cliff-walking task in Example 6.6 of the [textbook](http://go.udacity.com/rl-textbook). When you have finished, you can learn more about the environment in its corresponding [GitHub file](https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py), by reading the commented block in the CliffWalkingEnv class. For clarity, we have also pasted the description of the environment below (note that the link below to the Sutton and Barto textbook may not work, and you're encouraged to use [this link](http://go.udacity.com/rl-textbook) to access the textbook):

```
    """
    This is a simple implementation of the Gridworld Cliff
    reinforcement learning task.
    Adapted from Example 6.6 from Reinforcement Learning: An Introduction
    by Sutton and Barto:
    http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf

    With inspiration from:
    https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py
    The board is a 4x12 matrix, with (using Numpy matrix indexing):
        [3, 0] as the start at bottom-left
        [3, 11] as the goal at bottom-right
        [3, 1..10] as the cliff at bottom-center
    Each time step incurs -1 reward, and stepping into the cliff incurs -100 reward 
    and a reset to the start. An episode terminates when the agent reaches the goal.
    """
``` 


# Analyzing Performance

You've learned about three different TD control methods in this lesson. _So, what do they have in common, and how are they different?_

## Similarities

* * *

All of the TD control methods we have examined (Sarsa, Sarsamax, Expected Sarsa) converge to the optimal action-value function q∗q\_\*q∗​ (and so yield the optimal policy π∗\\pi\_\*π∗​) if:

1. the value of ϵ\\epsilonϵ decays in accordance with the GLIE conditions, and
2. the step-size parameter α\\alphaα is sufficiently small.

## Differences

* * *

The differences between these algorithms are summarized below:

- Sarsa and Expected Sarsa are both **on-policy** TD control algorithms. In this case, the same (ϵ\\epsilonϵ\-greedy) policy that is evaluated and improved is also used to select actions.
- Sarsamax is an **off-policy** method, where the (greedy) policy that is evaluated and improved is different from the (ϵ\\epsilonϵ\-greedy) policy that is used to select actions.
- On-policy TD control methods (like Expected Sarsa and Sarsa) have better online performance than off-policy TD control methods (like Sarsamax).
- Expected Sarsa generally achieves better performance than Sarsa.

If you would like to learn more, you are encouraged to read Chapter 6 of the [textbook](http://go.udacity.com/rl-textbook) (especially sections 6.4-6.6).

As an optional exercise to deepen your understanding, you are encouraged to reproduce Figure 6.4. (Note that this exercise is optional!)

![](https://video.udacity-data.com/topher/2018/May/5ae93d8e_screen-shot-2017-12-17-at-12.49.34-pm/screen-shot-2017-12-17-at-12.49.34-pm.png)

The figure shows the performance of Sarsa and Q-learning on the cliff walking environment for constant ϵ\=0.1\\epsilon = 0.1ϵ\=0.1. As described in the textbook, in this case,

- Q-learning achieves worse online performance (where the agent collects less reward on average in each episode), but learns the optimal policy, and
- Sarsa achieves better online performance, but learns a sub-optimal "safe" policy.

You should be able to reproduce the figure by making only small modifications to your existing code.


# Quiz: Check Your Understanding

In this lesson, you learned about many different algorithms for Temporal-Difference (TD) control. Later in this nanodegree, you'll learn more about how to adapt the Q-Learning algorithm to produce the Deep Q-Learning algorithm that demonstrated [superhuman performance](https://www.youtube.com/watch?v=V1eYniJ0Rnk) at Atari games.

Before moving on, you're encouraged to check your understanding by completing this brief quiz on **Q-Learning**.

![The Agent and Environment](https://video.udacity-data.com/topher/2018/March/5aa05f61_screen-shot-2018-03-07-at-3.53.08-pm/screen-shot-2018-03-07-at-3.53.08-pm.png)

The Agent and Environment

## The Agent and Environment

* * *

Imagine an agent that moves along a line with only five discrete positions (0, 1, 2, 3, or 4). The agent can move left, right or stay put. (_If the agent chooses to move left when at position 0 or right at position 4, the agent just remains in place._)

The Q-table has:

- five rows, corresponding to the five possible states that may be observed, and
- three columns, corresponding to three possible actions that the agent can take in response.

The goal state is position 3, but the agent doesn't know that and is going to learn the best policy for getting to the goal via the Q-Learning algorithm (with learning rate α\=0.2\\alpha=0.2α\=0.2). The environment will provide a reward of -1 for all locations except the goal state. The episode ends when the goal is reached.

## Episode 0, Time 0

* * *

The Q-table is initialized.

![Episode 0, Time 0](https://video.udacity-data.com/topher/2018/March/5aa04cad_screen-shot-2018-03-07-at-2.33.19-pm/screen-shot-2018-03-07-at-2.33.19-pm.png)

Episode 0, Time 0

Say the agent observes the initial **state** (position 1) and selects **action** stay.

As a result, it receives the **next state** (position 1) and a **reward** (-1.0) from the environment.

Let:

- sts\_tst​ denote the state at time step ttt,
- ata\_tat​ denote the action at time step ttt, and
- rtr\_trt​ denote the reward at time step ttt.

Then, the agent now knows s0,a0,r1s\_0, a\_0,r\_1s0​,a0​,r1​ and s1s\_1s1​. Namely, s0\=1,a0\=stay,r1\=−1.0,s\_0 = 1, a\_0=\\text{stay},r\_1=-1.0,s0​\=1,a0​\=stay,r1​\=−1.0, and s1\=1s\_1=1s1​\=1.

Using this information, it can update the Q-table value for Q(s0,a0)Q(s\_0, a\_0)Q(s0​,a0​). Recall the equation for updating the Q-table:

Q(st,at)←(1−α)⋅Q(st,at)⎵old value+α⎵learning rate⋅(rt+1⎵reward+γ⎵discount factor⋅maxaQ(st+1,a)⎵estimate of optimal future value)⏞learned value{\\displaystyle Q(s\_{t},a\_{t})\\leftarrow (1-\\alpha )\\cdot \\underbrace {Q(s\_{t},a\_{t})} \_{\\rm {old~value}}+\\underbrace {\\alpha } \_{\\rm {learning~rate}}\\cdot \\overbrace {{\\bigg (}\\underbrace {r\_{t+1}} \_{\\rm {reward}}+\\underbrace {\\gamma } \_{\\rm {discount~factor}}\\cdot \\underbrace {\\max \_{a}Q(s\_{t+1},a)} \_{\\rm {estimate~of~optimal~future~value}}{\\bigg )}} ^{\\rm {learned~value}}} Q(st​,at​)←(1−α)⋅old value Q(st​,at​)​​+learning rate α​​⋅(reward rt+1​​​+discount factor γ​​⋅estimate of optimal future value amax​Q(st+1​,a)​​) ​learned value​

Note that this is equivalent to the equation below (from the video on **Q-Learning**): Q(st,at)←Q(st,at)+α(rt+1+γmaxaQ(st+1,a)−Q(st,at))Q(s\_t, a\_t) \\leftarrow Q(s\_t, a\_t) + \\alpha(r\_{t+1} + \\gamma \\max\_a Q(s\_{t+1}, a) - Q(s\_t, a\_t)) Q(st​,at​)←Q(st​,at​)+α(rt+1​+γamax​Q(st+1​,a)−Q(st​,at​))

So the equation for updating Q(s0,a0)Q(s\_0, a\_0)Q(s0​,a0​) is: Q(s0,a0)←(1−α)⋅Q(s0,a0)+α⋅(r1+γmaxaQ(s1,a)) Q(s\_{0},a\_{0})\\leftarrow (1-\\alpha )\\cdot Q(s\_{0},a\_{0}) + \\alpha \\cdot (r\_1 + \\gamma \\max\_aQ(s\_1,a)) Q(s0​,a0​)←(1−α)⋅Q(s0​,a0​)+α⋅(r1​+γamax​Q(s1​,a))

Substituting our known values:

Q(s0,a0)←(1−0.2)⋅Q(s0,a0)+0.2⋅(r1+maxaQ(s1,a)){\\displaystyle Q(s\_{0},a\_{0})\\leftarrow (1-0.2 )\\cdot {Q(s\_{0},a\_{0})} +0.2\\cdot {{\\bigg (} {r\_{1}} +{\\max \_{a}Q(s\_{1},a)} {\\bigg )}} } Q(s0​,a0​)←(1−0.2)⋅Q(s0​,a0​)+0.2⋅(r1​+amax​Q(s1​,a))

We can find the _old value_ for Q(s0,a0)Q(s\_{0},a\_{0})Q(s0​,a0​) by looking it up in the table for state s0\=1s\_{0}=1s0​\=1 and action a0\=staya\_{0}=staya0​\=stay which is a value of 0. To find the _estimate of the optimal future value_ maxaQ(s1,a)\\max \_{a}Q(s\_{1},a)maxa​Q(s1​,a), we need to look at the entire row of actions for the _next_ state, s1\=1s\_{1}=1s1​\=1 and choose the maximum value across all actions. They are all 0 right now, so the maximum is 0. Reducing the equation, we can now update Q(s0,a0)Q(s\_{0},a\_{0})Q(s0​,a0​).

Q(s0,a0)←−0.2{\\displaystyle Q(s\_{0},a\_{0})\\leftarrow -0.2 } Q(s0​,a0​)←−0.2

## Episode 0, Time 1

* * *

![](https://video.udacity-data.com/topher/2018/March/5aa04efb_screen-shot-2018-03-07-at-2.43.07-pm/screen-shot-2018-03-07-at-2.43.07-pm.png)

At this step, an action must be chosen. The best action for position 1 could be either "left" or "right", since their values in the Q-table are equal.

Remember that in Q-Learning, the agent uses the epsilon-greedy policy to select an action. Say that in this case, the agent selects **action** right at random.

Then, the agent receives a **new state** (position 2) and **reward** (-1.0) from the environment.

The agent now knows s1,a1,r2,s\_1, a\_1,r\_2,s1​,a1​,r2​, and s2s\_2s2​.

## Episode n

* * *

Now assume that a number of episodes have been run, and the Q-table includes the values shown below.

A new episode begins, as before. The environment gives an initial **state** (position 1), and the agent selects **action** stay.

![Episode n, Time 0](https://video.udacity-data.com/topher/2018/March/5aa056d7_screen-shot-2018-03-07-at-3.16.47-pm/screen-shot-2018-03-07-at-3.16.47-pm.png)

Episode n, Time 0

# Summary

![The cliff-walking task (Sutton and Barto, 2017)](https://video.udacity-data.com/topher/2018/May/5ae93c67_screen-shot-2017-10-17-at-11.02.44-am/screen-shot-2017-10-17-at-11.02.44-am.png)

The cliff-walking task (Sutton and Barto, 2017)

### Temporal-Difference Methods

* * *

- Whereas Monte Carlo (MC) prediction methods must wait until the end of an episode to update the value function estimate, temporal-difference (TD) methods update the value function after every time step.

### TD Control

* * *

- **Sarsa(0)** (or **Sarsa**) is an on-policy TD control method. It is guaranteed to converge to the optimal action-value function q∗q\_\*q∗​, as long as the step-size parameter α\\alphaα is sufficiently small and ϵ\\epsilonϵ is chosen to satisfy the **Greedy in the Limit with Infinite Exploration (GLIE)** conditions.
- **Sarsamax** (or **Q-Learning**) is an off-policy TD control method. It is guaranteed to converge to the optimal action value function q∗q\_\*q∗​, under the same conditions that guarantee convergence of the Sarsa control algorithm.
- **Expected Sarsa** is an on-policy TD control method. It is guaranteed to converge to the optimal action value function q∗q\_\*q∗​, under the same conditions that guarantee convergence of Sarsa and Sarsamax.

### Analyzing Performance

* * *

- On-policy TD control methods (like Expected Sarsa and Sarsa) have better online performance than off-policy TD control methods (like Q-learning).
- Expected Sarsa generally achieves better performance than Sarsa.

