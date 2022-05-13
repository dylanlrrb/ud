## Review Your Notes

* * *

In the lesson **The RL Framework: The Problem**, you learned how to take a real-world problem and specify it in the language of reinforcement learning. In order to rigorously define a reinforcement learning task, we generally use a **Markov Decision Process (MDP)** to model the environment. The MDP specifies the rules that the environment uses to respond to the agent's actions, including how much reward to give to the agent in response to its behavior. The agent's goal is to learn how to play by the rules of the environment, in order to maximize reward.

![](https://video.udacity-data.com/topher/2017/September/59c29f47_screen-shot-2017-09-20-at-12.02.06-pm/screen-shot-2017-09-20-at-12.02.06-pm.png)

Agent-Environment Interaction

Next, in the lesson **The RL Framework: The Solution**, you learned how to specify a solution to the reinforcement learning problem. In particular, the **optimal policy** π∗\\pi\_\*π∗​ specifies - for each environment state - how the agent should select an action towards its goal of maximizing reward. You learned that the agent could structure its search for an optimal policy by first estimating the **optimal action-value function** q∗q\_\*q∗​; then, once q∗q\_\*q∗​ is known, π∗\\pi\_\*π∗​ is quickly obtained.

![Value Function](https://video.udacity-data.com/topher/2017/September/59c930a9_screen-shot-2017-09-25-at-11.35.38-am/screen-shot-2017-09-25-at-11.35.38-am.png)

Value Function

Before continuing with this lesson, please take the time to review your notes, to ensure that the terminology from the previous two lessons is familiar to you. In particular, you should peruse the summary page at the end of the lesson **The RL Framework: The Problem**, and the page at the end of **The RL Framework: The Solution** to ensure that the listed concepts are familiar.


# MC Prediction

![Coin flip ](https://video.udacity-data.com/topher/2018/April/5ae7362d_screen-shot-2018-04-30-at-10.27.56-am/screen-shot-2018-04-30-at-10.27.56-am.png)

So far in this lesson, we have discussed how the agent can take a bad policy, like the equiprobable random policy, use it to collect some episodes, and then consolidate the results to arrive at a better policy.

In the video in the previous concept, you saw that estimating the action-value function with a Q-table is an important intermediate step. We also refer to this as the **prediction problem**.

> **Prediction Problem**: Given a policy, how might the agent estimate the value function for that policy?

We've been specifically interested in the action-value function, but the **prediction problem** also refers to approaches that can be used to estimate the state-value function. We refer to Monte Carlo (MC) approaches to the prediction problem as **MC prediction methods**.

## Pseudocode

* * *

As you have learned in the videos, in the algorithm for MC prediction, we begin by collecting many episodes with the policy. Then, we note that each entry in the Q-table corresponds to a particular state and action. To populate an entry, we use the return that followed when the agent was in that state, and chose the action. In the event that the agent has selected the same action many times from the same state, we need only average the returns.

Before we dig into the pseudocode, we note that there are two different versions of MC prediction, depending on how you decide to treat the special case where - _in a single episode_ - the same action is selected from the same state many times. For more information, watch the video below.

As discussed in the video, we define every occurrence of a state in an episode as a **visit** to that state-action pair. And, in the event that a state-action pair is visited more than once in an episode, we have two options.

#### Option 1: Every-visit MC Prediction

Average the returns following all visits to each state-action pair, in all episodes.

#### Option 2: First-visit MC Prediction

For each episode, we only consider the first visit to the state-action pair. The pseudocode for this option can be found below.

![](https://video.udacity-data.com/topher/2018/May/5aecb9fe_screen-shot-2018-05-04-at-2.51.59-pm/screen-shot-2018-05-04-at-2.51.59-pm.png)

Don't let this pseudocode scare you! The main idea is quite simple. There are three relevant tables:

- QQQ - Q-table, with a row for each state and a column for each action. The entry corresponding to state sss and action aaa is denoted Q(s,a)Q(s,a)Q(s,a).
- NNN - table that keeps track of the number of first visits we have made to each state-action pair.
- returns\_sumreturns\\\_sumreturns\_sum - table that keeps track of the sum of the rewards obtained after first visits to each state-action pair.

In the algorithm, the number of episodes the agent collects is equal to num\_episodesnum\\\_episodesnum\_episodes. After each episode, NNN and returns\_sumreturns\\\_sumreturns\_sum are updated to store the information contained in the episode. Then, after all of the episodes have been collected and the values in NNN and returns\_sumreturns\\\_sumreturns\_sum have been finalized, we quickly obtain the final estimate for QQQ.

Soon, you'll have the chance to implement this algorithm yourself!

You will apply your code to OpenAI Gym's BlackJack environment. Note that in the game of BlackJack, first-visit and every-visit MC return identical results!

## First-visit or Every-visit?

* * *

Both the first-visit and every-visit method are **guaranteed to converge** to the true action-value function, as the number of visits to each state-action pair approaches infinity. (_So, in other words, as long as the agent gets enough experience with each state-action pair, the value function estimate will be pretty close to the true value._) In the case of first-visit MC, convergence follows from the [Law of Large Numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers), and the details are covered in section 5.1 of the [textbook](http://go.udacity.com/rl-textbook).

If you are interested in learning more about the difference between first-visit and every-visit MC methods, you are encouraged to read Section 3 of [this paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.32.9278&rep=rep1&type=pdf). The results are summarized in Section 3.6. The authors show:

- Every-visit MC is [biased](https://en.wikipedia.org/wiki/Bias_of_an_estimator), whereas first-visit MC is unbiased (see Theorems 6 and 7).
- Initially, every-visit MC has lower [mean squared error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error), but as more episodes are collected, first-visit MC attains better MSE (see Corollary 9a and 10a, and Figure 4).


# OpenAI Gym: BlackJackEnv

In order to master the algorithms discussed in this lesson, you will write code to teach an agent to play Blackjack.

![Playing Cards ([Source](https://www.blackjackinfo.com/img/2-card-21.png))](https://video.udacity-data.com/topher/2017/October/59d245d3_2-card-21/2-card-21.png)

Playing Cards ([Source](https://www.blackjackinfo.com/img/2-card-21.png))

Please read about the game of Blackjack in Example 5.1 of the [textbook](http://go.udacity.com/rl-textbook).

When you have finished, please review the corresponding [GitHub file](https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py), by reading the commented block in the BlackjackEnv class. (_While you do **not** need to understand how all of the code works, please read the commented block that explains the dynamics of the environment._) For clarity, we have also pasted the description of the environment below:

```
    """Simple blackjack environment

    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with each (player and dealer) having one face up and one
    face down card.

    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).

    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.

    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.

    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).

    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto (1998).
    http://incompleteideas.net/sutton/book/the-book.html
    """
``` 

## Note

* * *

In the video above, you learned about ϵ\\epsilonϵ\-greedy policies.

You can think of the agent who follows an ϵ\\epsilonϵ\-greedy policy as always having a (potentially unfair) coin at its disposal, with probability ϵ\\epsilonϵ of landing heads. After observing a state, the agent flips the coin.

- If the coin lands tails (so, with probability 1−ϵ1-\\epsilon1−ϵ), the agent selects the greedy action.
- If the coin lands heads (so, with probability ϵ\\epsilonϵ), the agent selects an action _uniformly_ at random from the set of available (non-greedy **AND** greedy) actions.

In order to construct a policy π\\piπ that is ϵ\\epsilonϵ\-greedy with respect to the current action-value function estimate QQQ, we will set

π(a∣s)⟵{1−ϵ+ϵ/∣A(s)∣if a maximizes Q(s,a)ϵ/∣A(s)∣else \\pi(a|s) \\longleftarrow \\begin{cases} \\displaystyle 1-\\epsilon +\\epsilon/|\\mathcal{A}(s)|& \\textrm{if }a\\textrm{ maximizes }Q(s,a)\\\\ \\displaystyle \\epsilon/|\\mathcal{A}(s)| & \\textrm{else} \\end{cases} π(a∣s)⟵{1−ϵ+ϵ/∣A(s)∣ϵ/∣A(s)∣​if a maximizes Q(s,a)else​

for each s∈Ss\\in\\mathcal{S}s∈S and a∈A(s)a\\in\\mathcal{A}(s)a∈A(s).

Mathematically, A(s)\\mathcal{A}(s)A(s) is the set of all possible actions at state sss (which may be 'up', 'down','right', 'left' for example), and ∣A(s)∣|\\mathcal{A}(s)|∣A(s)∣ the number of possible actions (including the optimal one!). The reason why we include an extra term ϵ/∣A(s)∣\\epsilon/|\\mathcal{A}(s)|ϵ/∣A(s)∣ for the optimal action is because the sum of all the probabilities needs to be 1. If we sum over the probabilities of performing non-optimal actions, we will get (∣A(s)∣−1)×ϵ/∣A(s)∣(|\\mathcal{A}(s)|-1)\\times \\epsilon/|\\mathcal{A}(s)| (∣A(s)∣−1)×ϵ/∣A(s)∣, and adding this to 1−ϵ+ϵ/∣A(s)∣1-\\epsilon + \\epsilon/|\\mathcal{A}(s)|1−ϵ+ϵ/∣A(s)∣ gives one.

Note that ϵ\\epsilonϵ must always be a value between 0 and 1, inclusive (that is, ϵ∈\[0,1\]\\epsilon \\in \[0,1\]ϵ∈\[0,1\]).

In this quiz, you will answer a few questions to test your intuition.


# MC Control

So far, you have learned how the agent can take a policy π\\piπ, use it to interact with the environment for many episodes, and then use the results to estimate the action-value function qπq\_\\piqπ​ with a Q-table.

Then, once the Q-table closely approximates the action-value function qπq\_\\piqπ​, the agent can construct the policy π′\\pi'π′ that is ϵ\\epsilonϵ\-greedy with respect to the Q-table, which will yield a policy that is better than the original policy π\\piπ.

Furthermore, if the agent alternates between these two steps, with:

- **Step 1**: using the policy π\\piπ to construct the Q-table, and
- **Step 2**: improving the policy by changing it to be ϵ\\epsilonϵ\-greedy with respect to the Q-table (π′←ϵ\-greedy(Q)\\pi' \\leftarrow \\epsilon\\text{-greedy}(Q)π′←ϵ\-greedy(Q), π←π′\\pi \\leftarrow \\pi'π←π′),

we will eventually obtain the optimal policy π∗\\pi\_\*π∗​.

Since this algorithm is a solution for the **control problem** (defined below), we call it a **Monte Carlo control method**.

> **Control Problem**: Estimate the optimal policy.

It is common to refer to **Step 1** as **policy evaluation**, since it is used to determine the action-**value** function of the policy. Likewise, since **Step 2** is used to **improve** the policy, we also refer to it as a **policy improvement** step.

![MC Control](https://video.udacity-data.com/topher/2018/May/5aedf602_screen-shot-2018-05-05-at-1.20.10-pm/screen-shot-2018-05-05-at-1.20.10-pm.png)

MC Control

So, using this new terminology, we can summarize what we've learned to say that our **Monte Carlo control method** alternates between **policy evaluation** and **policy improvement** steps to recover the optimal policy π∗\\pi\_\*π∗​.

## The Road Ahead

* * *

You now have a working algorithm for Monte Carlo control! So, what's to come?

- In the next concept (**Exploration vs. Exploitation**), you will learn more about how to set the value of ϵ\\epsilonϵ when constructing ϵ\\epsilonϵ\-greedy policies in the policy improvement step.
- Then, you will learn about two improvements that you can make to the policy evaluation step in your control algorithm.
    - In the **Incremental Mean** concept, you will learn how to update the policy after every episode (instead of waiting to update the policy until after the values of the Q-table have fully converged from many episodes).
    - In the **Constant-alpha** concept, you will learn how to train the agent to leverage its most recent experience more effectively.

Finally, to conclude the lesson, you will write your own algorithm for Monte Carlo control to solve OpenAI Gym's Blackjack environment, to put your new knowledge to practice!


# Exploration vs. Exploitation

![Exploration-Exploitation Dilemma](https://video.udacity-data.com/topher/2017/October/59d55ce3_exploration-vs.-exploitation/exploration-vs.-exploitation.png)

Exploration-Exploitation Dilemma ([Source](http://slides.com/ericmoura/deck-2/embed))

## Solving Environments in OpenAI Gym

* * *

In many cases, we would like our reinforcement learning (RL) agents to learn to maximize reward as quickly as possible. This can be seen in many OpenAI Gym environments.

For instance, the [FrozenLake-v0](https://gym.openai.com/envs/FrozenLake-v0/) environment is considered solved once the agent attains an average reward of 0.78 over 100 consecutive trials.

![](https://video.udacity-data.com/topher/2017/October/59d559bf_screen-shot-2017-10-04-at-4.58.58-pm/screen-shot-2017-10-04-at-4.58.58-pm.png)

Algorithmic solutions to the [FrozenLake-v0](https://gym.openai.com/envs/FrozenLake-v0/) environment are ranked according to the number of episodes needed to find the solution.

![](https://video.udacity-data.com/topher/2017/October/59d55a58_screen-shot-2017-10-04-at-5.01.26-pm/screen-shot-2017-10-04-at-5.01.26-pm.png)

Solutions to [Taxi-v1](https://gym.openai.com/envs/Taxi-v1/), [Cartpole-v1](https://gym.openai.com/envs/CartPole-v1/), and [MountainCar-v0](https://gym.openai.com/envs/MountainCar-v0/) (along with many others) are also ranked according to the number of episodes before the solution is found. Towards this objective, it makes sense to design an algorithm that learns the optimal policy π∗\\pi\_\*π∗​ as quickly as possible.

## Exploration-Exploitation Dilemma

* * *

Recall that the environment's dynamics are initially unknown to the agent. Towards maximizing return, the agent must learn about the environment through interaction.

At every time step, when the agent selects an action, it bases its decision on past experience with the environment. And, towards minimizing the number of episodes needed to solve environments in OpenAI Gym, our first instinct could be to devise a strategy where the agent always selects the action that it believes (_based on its past experience_) will maximize return. With this in mind, the agent could follow the policy that is greedy with respect to the action-value function estimate. We examined this approach in a previous video and saw that it can easily lead to convergence to a sub-optimal policy.

To see why this is the case, note that in early episodes, the agent's knowledge is quite limited (and potentially flawed). So, it is highly likely that actions _estimated_ to be non-greedy by the agent are in fact better than the _estimated_ greedy action.

With this in mind, a successful RL agent cannot act greedily at every time step (_that is_, it cannot always **exploit** its knowledge); instead, in order to discover the optimal policy, it has to continue to refine the estimated return for all state-action pairs (_in other words_, it has to continue to **explore** the range of possibilities by visiting every state-action pair). That said, the agent should always act _somewhat greedily_, towards its goal of maximizing return _as quickly as possible_. This motivated the idea of an ϵ\\epsilonϵ\-greedy policy.

We refer to the need to balance these two competing requirements as the **Exploration-Exploitation Dilemma**. One potential solution to this dilemma is implemented by gradually modifying the value of ϵ\\epsilonϵ when constructing ϵ\\epsilonϵ\-greedy policies.

## Setting the Value of ϵ\\epsilonϵ, in Theory

* * *

It makes sense for the agent to begin its interaction with the environment by favoring **exploration** over **exploitation**. After all, when the agent knows relatively little about the environment's dynamics, it should distrust its limited knowledge and **explore**, or try out various strategies for maximizing return. With this in mind, the best starting policy is the equiprobable random policy, as it is equally likely to explore all possible actions from each state. You discovered in the previous quiz that setting ϵ\=1\\epsilon = 1ϵ\=1 yields an ϵ\\epsilonϵ\-greedy policy that is equivalent to the equiprobable random policy.

At later time steps, it makes sense to favor **exploitation** over **exploration**, where the policy gradually becomes more greedy with respect to the action-value function estimate. After all, the more the agent interacts with the environment, the more it can trust its estimated action-value function. You discovered in the previous quiz that setting ϵ\=0\\epsilon = 0ϵ\=0 yields the greedy policy (or, the policy that most favors exploitation over exploration).

Thankfully, this strategy (of initially favoring exploration over exploitation, and then gradually preferring exploitation over exploration) can be demonstrated to be optimal.

## Greedy in the Limit with Infinite Exploration (GLIE)

* * *

In order to guarantee that MC control converges to the optimal policy π∗\\pi\_\*π∗​, we need to ensure that two conditions are met. We refer to these conditions as **Greedy in the Limit with Infinite Exploration (GLIE)**. In particular, if:

- every state-action pair s,as, as,a (for all s∈Ss\\in\\mathcal{S}s∈S and a∈A(s)a\\in\\mathcal{A}(s)a∈A(s)) is visited infinitely many times, and
- the policy converges to a policy that is greedy with respect to the action-value function estimate QQQ,

then MC control is guaranteed to converge to the optimal policy (in the limit as the algorithm is run for infinitely many episodes). These conditions ensure that:

- the agent continues to explore for all time steps, and
- the agent gradually exploits more (and explores less).

One way to satisfy these conditions is to modify the value of ϵ\\epsilonϵ when specifying an ϵ\\epsilonϵ\-greedy policy. In particular, let ϵi\\epsilon\_iϵi​ correspond to the iii\-th time step. Then, both of these conditions are met if:

- ϵi\>0\\epsilon\_i > 0ϵi​\>0 for all time steps iii, and
- ϵi\\epsilon\_iϵi​ decays to zero in the limit as the time step iii approaches infinity (that is, limi→∞ϵi\=0\\lim\_{i\\to\\infty} \\epsilon\_i = 0limi→∞​ϵi​\=0).

For example, to ensure convergence to the optimal policy, we could set ϵi\=1i\\epsilon\_i = \\frac{1}{i}ϵi​\=i1​. (You are encouraged to verify that ϵi\>0\\epsilon\_i > 0ϵi​\>0 for all iii, and limi→∞ϵi\=0\\lim\_{i\\to\\infty} \\epsilon\_i = 0limi→∞​ϵi​\=0.)

## Setting the Value of ϵ\\epsilonϵ, in Practice

* * *

As you read in the above section, in order to guarantee convergence, we must let ϵi\\epsilon\_iϵi​ decay in accordance with the GLIE conditions. But sometimes "guaranteed convergence" _isn't good enough_ in practice, since this really doesn't tell you how long you have to wait! It is possible that you could need trillions of episodes to recover the optimal policy, for instance, and the "guaranteed convergence" would still be accurate!

> Even though convergence is **not** guaranteed by the mathematics, you can often get better results by either:
> 
> - using fixed ϵ\\epsilonϵ, or
> - letting ϵi\\epsilon\_iϵi​ decay to a small positive number, like 0.1.

This is because one has to be very careful with setting the decay rate for ϵ\\epsilonϵ; letting it get too small too fast can be disastrous. If you get late in training and ϵ\\epsilonϵ is really small, you pretty much want the agent to have already converged to the optimal policy, as it will take way too long otherwise for it to test out new actions!

As a famous example in practice, you can read more about how the value of ϵ\\epsilonϵ was set in the famous DQN algorithm by reading the Methods section of [the research paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf):

> _The behavior policy during training was epsilon-greedy with epsilon annealed linearly from 1.0 to 0.1 over the first million frames, and fixed at 0.1 thereafter._

When you implement your own algorithm for MC control later in this lesson, you are strongly encouraged to experiment with setting the value of ϵ\\epsilonϵ to build your intuition.


# Incremental Mean

In our current algorithm for Monte Carlo control, we collect a large number of episodes to build the Q-table (as an estimate for the action-value function corresponding to the agent's current policy). Then, after the values in the Q-table have converged, we use the table to come up with an improved policy.

Maybe it would be more efficient to update the Q-table **_after every episode_**. Then, the updated Q-table could be used to improve the policy. That new policy could then be used to generate the next episode, and so on.

![MC Control with Incremental Mean](https://video.udacity-data.com/topher/2018/May/5af4d181_screen-shot-2018-05-10-at-6.10.16-pm/screen-shot-2018-05-10-at-6.10.16-pm.png)

MC Control with Incremental Mean

So, _how might we modify our code to accomplish this_? Watch the video below to see!


In this case, even though we're updating the policy before the values in the Q-table accurately approximate the action-value function, this lower-quality estimate nevertheless still has enough information to help us propose successively better policies. If you're curious to learn more, you can read section 5.6 of [the textbook](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/suttonbookdraft2018jan1.pdf).

## Pseudocode

* * *

The pseudocode can be found below.

![](https://video.udacity-data.com/topher/2018/May/5aecc149_screen-shot-2018-05-04-at-3.14.47-pm/screen-shot-2018-05-04-at-3.14.47-pm.png)

There are two relevant tables:

- QQQ - Q-table, with a row for each state and a column for each action. The entry corresponding to state sss and action aaa is denoted Q(s,a)Q(s,a)Q(s,a).
- NNN - table that keeps track of the number of first visits we have made to each state-action pair.

The number of episodes the agent collects is equal to num\_episodesnum\\\_episodesnum\_episodes.

The algorithm proceeds by looping over the following steps:

- **Step 1**: The policy π\\piπ is improved to be ϵ\\epsilonϵ\-greedy with respect to QQQ, and the agent uses π\\piπ to collect an episode.
- **Step 2**: NNN is updated to count the total number of first visits to each state action pair.
- **Step 3**: The estimates in QQQ are updated to take into account the most recent information.

In this way, the agent is able to improve the policy after every episode!


# Constant-alpha

In the video below, you will learn about another improvement that you can make to your Monte Carlo control algorithm.

## Pseudocode

* * *

The pseudocode for constant-α\\alphaα GLIE MC Control can be found below.

![](https://video.udacity-data.com/topher/2018/May/5aecba4c_screen-shot-2018-05-04-at-2.49.48-pm/screen-shot-2018-05-04-at-2.49.48-pm.png)

## Setting the Value of α\\alphaα

* * *

Recall the update equation that we use to amend the values in the Q-table: Q(St,At)←Q(St,At)+α(Gt−Q(St,At))Q(S\_t, A\_t) \\leftarrow Q(S\_t, A\_t) + \\alpha (G\_t - Q(S\_t, A\_t))Q(St​,At​)←Q(St​,At​)+α(Gt​−Q(St​,At​))

To examine how to set the the value of α\\alphaα in more detail, we will slightly rewrite the equation as follows:

Q(St,At)←(1−α)Q(St,At)+αGtQ(S\_t,A\_t) \\leftarrow (1-\\alpha)Q(S\_t,A\_t) + \\alpha G\_tQ(St​,At​)←(1−α)Q(St​,At​)+αGt​

Watch the video below to hear more about how to set the value of α\\alphaα.

Here are some guiding principles that will help you to set the value of α\\alphaα when implementing constant-α\\alphaα MC control:

- You should always set the value for α\\alphaα to a number greater than zero and less than (or equal to) one.
    
    - If α\=0\\alpha=0α\=0, then the action-value function estimate is never updated by the agent.
    - If α\=1\\alpha = 1α\=1, then the final value estimate for each state-action pair is always equal to the last return that was experienced by the agent (after visiting the pair).
- Smaller values for α\\alphaα encourage the agent to consider a longer history of returns when calculating the action-value function estimate. Increasing the value of α\\alphaα ensures that the agent focuses more on the most recently sampled returns.
    

> **Important Note**: When implementing constant-α\\alphaα MC control, you must be careful to not set the value of α\\alphaα too close to 1. This is because very large values can keep the algorithm from converging to the optimal policy π∗\\pi\_\*π∗​. However, you must also be careful to not set the value of α\\alphaα too low, as this can result in an agent who learns too slowly. The best value of α\\alphaα for your implementation will greatly depend on your environment and is best gauged through trial-and-error.


# Summary

![](https://video.udacity-data.com/topher/2017/October/59d69c65_screen-shot-2017-10-05-at-3.55.40-pm/screen-shot-2017-10-05-at-3.55.40-pm.png)

Optimal Policy and State-Value Function in Blackjack (Sutton and Barto, 2017)

### Monte Carlo Methods

* * *

- Monte Carlo methods - even though the underlying problem involves a great degree of randomness, we can infer useful information that we can trust just by collecting a lot of samples.
- The **equiprobable random policy** is the stochastic policy where - from each state - the agent randomly selects from the set of available actions, and each action is selected with equal probability.

### MC Prediction

* * *

- Algorithms that solve the **prediction problem** determine the value function vπv\_\\pivπ​ (or qπq\_\\piqπ​) corresponding to a policy π\\piπ.
- When working with finite MDPs, we can estimate the action-value function qπq\_\\piqπ​ corresponding to a policy π\\piπ in a table known as a **Q-table**. This table has one row for each state and one column for each action. The entry in the sss\-th row and aaa\-th column contains the agent's estimate for expected return that is likely to follow, if the agent starts in state sss, selects action aaa, and then henceforth follows the policy π\\piπ.
- Each occurrence of the state-action pair s,as,as,a (s∈S,a∈As\\in\\mathcal{S},a\\in\\mathcal{A}s∈S,a∈A) in an episode is called a **visit to s,as,as,a**.
- There are two types of MC prediction methods (for estimating qπq\_\\piqπ​):
    - **First-visit MC** estimates qπ(s,a)q\_\\pi(s,a)qπ​(s,a) as the average of the returns following _only first_ visits to s,as,as,a (that is, it ignores returns that are associated to later visits).
    - **Every-visit MC** estimates qπ(s,a)q\_\\pi(s,a)qπ​(s,a) as the average of the returns following _all_ visits to s,as,as,a.

### Greedy Policies

* * *

- A policy is **greedy** with respect to an action-value function estimate QQQ if for every state s∈Ss\\in\\mathcal{S}s∈S, it is guaranteed to select an action a∈A(s)a\\in\\mathcal{A}(s)a∈A(s) such that a\=argmaxa∈A(s)Q(s,a)a = \\arg\\max\_{a\\in\\mathcal{A}(s)}Q(s,a)a\=argmaxa∈A(s)​Q(s,a). (It is common to refer to the selected action as the **greedy action**.)
- In the case of a finite MDP, the action-value function estimate is represented in a Q-table. Then, to get the greedy action(s), for each row in the table, we need only select the action (or actions) corresponding to the column(s) that maximize the row.

### Epsilon-Greedy Policies

* * *

- A policy is **ϵ\\epsilonϵ\-greedy** with respect to an action-value function estimate QQQ if for every state s∈Ss\\in\\mathcal{S}s∈S,
    - with probability 1−ϵ1-\\epsilon1−ϵ, the agent selects the greedy action, and
    - with probability ϵ\\epsilonϵ, the agent selects an action _uniformly_ at random from the set of available (non-greedy **AND** greedy) actions.

### MC Control

* * *

- Algorithms designed to solve the **control problem** determine the optimal policy π∗\\pi\_\*π∗​ from interaction with the environment.
- The **Monte Carlo control method** uses alternating rounds of policy evaluation and improvement to recover the optimal policy.

### Exploration vs. Exploitation

* * *

- All reinforcement learning agents face the **Exploration-Exploitation Dilemma**, where they must find a way to balance the drive to behave optimally based on their current knowledge (**exploitation**) and the need to acquire knowledge to attain better judgment (**exploration**).
- In order for MC control to converge to the optimal policy, the **Greedy in the Limit with Infinite Exploration (GLIE)** conditions must be met:
    - every state-action pair s,as, as,a (for all s∈Ss\\in\\mathcal{S}s∈S and a∈A(s)a\\in\\mathcal{A}(s)a∈A(s)) is visited infinitely many times, and
    - the policy converges to a policy that is greedy with respect to the action-value function estimate QQQ.

### Incremental Mean

* * *

- (In this concept, we amended the policy evaluation step to update the Q-table after every episode of interaction.)

### Constant-alpha

* * *

- (In this concept, we derived the algorithm for **constant-α\\alphaα MC control**, which uses a constant step-size parameter α\\alphaα.)
- The step-size parameter α\\alphaα must satisfy 0<α≤10 < \\alpha \\leq 10<α≤1. Higher values of α\\alphaα will result in faster learning, but values of α\\alphaα that are too high can prevent MC control from converging to π∗\\pi\_\*π∗​.


