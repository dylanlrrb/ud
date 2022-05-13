This lesson covers material in **Chapter 3** (especially 3.1-3.3) of the [textbook](http://go.udacity.com/rl-textbook).

# Quiz: Episodic or Continuing?

Remember:

- A **task** is an instance of the reinforcement learning (RL) problem.
- **Continuing tasks** are tasks that continue forever, without end.
- **Episodic tasks** are tasks with a well-defined starting and ending point.
    - In this case, we refer to a complete sequence of interaction, from start to finish, as an **episode**.
    - Episodic tasks come to an end whenever the agent reaches a **terminal state**.

With these ideas in mind, use the quiz below to classify tasks as continuing or episodic.

If you'd like to learn more about the research that was done at [DeepMind](https://deepmind.com/), please check out [this link](https://deepmind.com/blog/producing-flexible-behaviours-simulated-environments/). The research paper can be accessed [here](https://arxiv.org/pdf/1707.02286.pdf). Also, check out this cool [video](https://www.youtube.com/watch?v=hx_bgoTF7bs&feature=youtu.be)!


**Note**: In this course, we will use "return" and "discounted return" interchangably. For an arbitrary time step ttt, both refer to Gt≐Rt+1+γRt+2+γ2Rt+3+…\=∑k\=0∞γkRt+k+1G\_t \\doteq R\_{t+1} + \\gamma R\_{t+2} + \\gamma^2 R\_{t+3} + \\ldots = \\sum\_{k=0}^\\infty \\gamma^k R\_{t+k+1}Gt​≐Rt+1​+γRt+2​+γ2Rt+3​+…\=∑k\=0∞​γkRt+k+1​, where γ∈\[0,1\]\\gamma \\in \[0,1\]γ∈\[0,1\]. In particular, when we refer to "return", it is not necessarily the case that γ\=1\\gamma = 1γ\=1, and when we refer to "discounted return", it is not necessarily true that γ<1\\gamma < 1γ<1. (_This also holds for the readings in the recommended textbook._)


# Quiz: Pole-Balancing

![](https://video.udacity-data.com/topher/2017/September/59c3402c_1omsg2-mkguagky1c64uflw/1omsg2-mkguagky1c64uflw.gif)

Source: [https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947](https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947)

In this classic reinforcement learning task, a cart is positioned on a frictionless track, and a pole is attached to the top of the cart. The objective is to keep the pole from falling over by moving the cart either left or right, and without falling off the track.

In the [OpenAI Gym implementation](https://gym.openai.com/envs/CartPole-v0/), the agent applies a force of +1 or -1 to the cart at every time step. It is formulated as an episodic task, where the episode ends when (1) the pole falls more than 20.9 degrees from vertical, (2) the cart moves more than 2.4 units from the center of the track, or (3) when more than 200 time steps have elapsed. The agent receives a reward of +1 for every time step, including the final step of the episode. You can read more about this environment in [OpenAI's github](https://github.com/openai/gym/wiki/CartPole-v0). This task also appears in Example 3.4 of the textbook.


# MDPs

Over the next several videos, you'll learn all about how to rigorously define a reinforcement learning problem as a **Markov Decision Process (MDP)**.

Towards this goal, we'll begin with an example!

## Notes

* * *

In general, the state space S\\mathcal{S}S is the set of **all nonterminal states**.

In continuing tasks (like the recycling task detailed in the video), this is equivalent to the set of **all states**.

In episodic tasks, we use S+\\mathcal{S}^+S+ to refer to the set of **all states, including terminal states**.

The action space A\\mathcal{A}A is the set of possible actions available to the agent.

In the event that there are some states where only a subset of the actions are available, we can also use A(s)\\mathcal{A}(s)A(s) to refer to the set of actions available in state s∈Ss\\in\\mathcal{S}s∈S.


# Quiz: One-Step Dynamics

Consider the recycling robot example. In the previous concept, we described one method that the environment could use to decide the state and reward, at any time step.

![](https://video.udacity-data.com/topher/2017/September/59c3f51a_screen-shot-2017-09-21-at-12.20.30-pm/screen-shot-2017-09-21-at-12.20.30-pm.png)

Say at an arbitrary time step ttt, the state of the robot's battery is high (St\=highS\_t = \\text{high}St​\=high). Then, in response, the agent decides to search (At\=searchA\_t = \\text{search}At​\=search). You learned in the previous concept that in this case, the environment responds to the agent by flipping a theoretical coin with 70% probability of landing heads.

- If the coin lands heads, the environment decides that the next state is high (St+1\=highS\_{t+1} = \\text{high}St+1​\=high), and the reward is 4 (Rt+1\=4R\_{t+1} = 4Rt+1​\=4).
- If the coin lands tails, the environment decides that the next state is low (St+1\=lowS\_{t+1} = \\text{low}St+1​\=low), and the reward is 4 (Rt+1\=4R\_{t+1} = 4Rt+1​\=4).

This is depicted in the figure below.

![](https://video.udacity-data.com/topher/2017/September/59c3f529_screen-shot-2017-09-21-at-12.20.50-pm/screen-shot-2017-09-21-at-12.20.50-pm.png)

In fact, for any state StS\_{t}St​ and action AtA\_{t}At​, it is possible to use the figure to determine exactly how the agent will decide the next state St+1S\_{t+1}St+1​ and reward Rt+1R\_{t+1}Rt+1​.


# Quiz: One-Step Dynamics

It will prove convenient to represent the environment's dynamics using mathematical notation. In this concept, we will introduce this notation (which can be used for any reinforcement learning task) and use the recycling robot as an example.

![](https://video.udacity-data.com/topher/2017/September/59c40b85_screen-shot-2017-09-21-at-12.20.30-pm/screen-shot-2017-09-21-at-12.20.30-pm.png)

At an arbitrary time step ttt, the agent-environment interaction has evolved as a sequence of states, actions, and rewards

(S0,A0,R1,S1,A1,…,Rt−1,St−1,At−1,Rt,St,At)(S\_0, A\_0, R\_1, S\_1, A\_1, \\ldots, R\_{t-1}, S\_{t-1}, A\_{t-1}, R\_t, S\_t, A\_t)(S0​,A0​,R1​,S1​,A1​,…,Rt−1​,St−1​,At−1​,Rt​,St​,At​).

When the environment responds to the agent at time step t+1t+1t+1, it considers only the state and action at the previous time step (St,AtS\_t, A\_tSt​,At​).

In particular, it does not care what state was presented to the agent more than one step prior. (_In other words_, the environment does not consider any of {S0,…,St−1}\\{ S\_0, \\ldots, S\_{t-1} \\}{S0​,…,St−1​}.)

And, it does not look at the actions that the agent took prior to the last one. (_In other words_, the environment does not consider any of {A0,…,At−1}\\{ A\_0, \\ldots, A\_{t-1} \\}{A0​,…,At−1​}.)

Furthermore, how well the agent is doing, or how much reward it is collecting, has no effect on how the environment chooses to respond to the agent. (_In other words_, the environment does not consider any of {R0,…,Rt}\\{ R\_0, \\ldots, R\_t \\} {R0​,…,Rt​}.)

Because of this, we can completely define how the environment decides the state and reward by specifying

p(s′,r∣s,a)≐P(St+1\=s′,Rt+1\=r∣St\=s,At\=a)p(s',r|s,a) \\doteq \\mathbb{P}(S\_{t+1}=s', R\_{t+1}=r|S\_t = s, A\_t=a)p(s′,r∣s,a)≐P(St+1​\=s′,Rt+1​\=r∣St​\=s,At​\=a)

for each possible s′,r,s,and as', r, s, \\text{and } as′,r,s,and a. These conditional probabilities are said to specify the **one-step dynamics** of the environment.

## An Example

Let's return to the case that St\=highS\_t = \\text{high}St​\=high, and At\=searchA\_t = \\text{search}At​\=search.

![](https://video.udacity-data.com/topher/2017/September/59c40bf7_screen-shot-2017-09-21-at-12.20.50-pm/screen-shot-2017-09-21-at-12.20.50-pm.png)

Then, when the environment responds to the agent at the next time step,

- with 70% probability, the next state is high and the reward is 4. In other words, p(high,4∣high,search)\=P(St+1\=high,Rt+1\=4∣St\=high,At\=search)\=0.7p(\\text{high}, 4|\\text{high},\\text{search}) = \\mathbb{P}(S\_{t+1}=\\text{high}, R\_{t+1}=4|S\_{t} = \\text{high}, A\_{t}=\\text{search}) = 0.7p(high,4∣high,search)\=P(St+1​\=high,Rt+1​\=4∣St​\=high,At​\=search)\=0.7.
    
- with 30% probability, the next state is low and the reward is 4. In other words, p(low,4∣high,search)\=P(St+1\=low,Rt+1\=4∣St\=high,At\=search)\=0.3p(\\text{low}, 4|\\text{high},\\text{search}) = \\mathbb{P}(S\_{t+1}=\\text{low}, R\_{t+1}=4|S\_{t} = \\text{high}, A\_{t}=\\text{search}) = 0.3p(low,4∣high,search)\=P(St+1​\=low,Rt+1​\=4∣St​\=high,At​\=search)\=0.3.


# Finite MDPs

Please use [this link](https://github.com/openai/gym/wiki/Table-of-environments) to peruse the available environments in OpenAI Gym.

![](https://video.udacity-data.com/topher/2017/September/59c41c74_screen-shot-2017-09-21-at-3.08.03-pm/screen-shot-2017-09-21-at-3.08.03-pm.png)

The environments are indexed by **Environment Id**, and each environment has corresponding **Observation Space**, **Action Space**, **Reward Range**, **tStepL**, **Trials**, and **rThresh**.

## CartPole-v0

* * *

Find the line in the table that corresponds to the **CartPole-v0** environment. Take note of the corresponding **Observation Space** (`Box(4,)`) and **Action Space** (`Discrete(2)`).

![](https://video.udacity-data.com/topher/2017/September/59c42093_screen-shot-2017-09-21-at-3.25.10-pm/screen-shot-2017-09-21-at-3.25.10-pm.png)

As described in the [OpenAI Gym documentation](https://gym.openai.com/docs/),

> Every environment comes with first-class `Space` objects that describe the valid actions and observations.
> 
> - The `Discrete` space allows a fixed range of non-negative numbers.
> - The `Box` space represents an n-dimensional box, so valid actions or observations will be an array of n numbers.

## Observation Space

* * *

The observation space for the CartPole-v0 environment has type `Box(4,)`. Thus, the observation (or state) at each time point is an array of 4 numbers. You can look up what each of these numbers represents in [this document](https://github.com/openai/gym/wiki/CartPole-v0). After opening the page, scroll down to the description of the observation space.

![](https://video.udacity-data.com/topher/2017/September/59c42575_screen-shot-2017-09-21-at-3.46.12-pm/screen-shot-2017-09-21-at-3.46.12-pm.png)

Notice the minimum (-Inf) and maximum (Inf) values for both **Cart Velocity** and the **Pole Velocity at Tip**.

Since the entry in the array corresponding to each of these indices can be any real number, the state space S+\\mathcal{S}^+S+ is infinite!

## Action Space

* * *

The action space for the CartPole-v0 environment has type `Discrete(2)`. Thus, at any time point, there are only two actions available to the agent. You can look up what each of these numbers represents in [this document](https://github.com/openai/gym/wiki/CartPole-v0) (note that it is the same document you used to look up the observation space!). After opening the page, scroll down to the description of the action space.

![](https://video.udacity-data.com/topher/2017/September/59c4305c_screen-shot-2017-09-21-at-4.34.08-pm/screen-shot-2017-09-21-at-4.34.08-pm.png)

In this case, the action space A\\mathcal{A}A is a finite set containing only two elements.

## Finite MDPs

* * *

Recall from the previous concept that in a finite MDP, the state space S\\mathcal{S}S (or S+\\mathcal{S}^+S+, in the case of an episodic task) and action space A\\mathcal{A}A must both be finite.

Thus, while the CartPole-v0 environment does specify an MDP, it does not specify a **finite** MDP. In this course, we will first learn how to solve finite MDPs. Then, later in this course, you will learn how to use neural networks to solve much more complex MDPs!



# Summary

![](https://video.udacity-data.com/topher/2017/September/59c29f47_screen-shot-2017-09-20-at-12.02.06-pm/screen-shot-2017-09-20-at-12.02.06-pm.png)

The agent-environment interaction in reinforcement learning. (Source: Sutton and Barto, 2017)

### The Setting, Revisited

* * *

- The reinforcement learning (RL) framework is characterized by an **agent** learning to interact with its **environment**.
- At each time step, the agent receives the environment's **state** (_the environment presents a situation to the agent)_, and the agent must choose an appropriate **action** in response. One time step later, the agent receives a **reward** (_the environment indicates whether the agent has responded appropriately to the state_) and a new **state**.
- All agents have the goal to maximize expected **cumulative reward**, or the expected sum of rewards attained over all time steps.

### Episodic vs. Continuing Tasks

* * *

- A **task** is an instance of the reinforcement learning (RL) problem.
- **Continuing tasks** are tasks that continue forever, without end.
- **Episodic tasks** are tasks with a well-defined starting and ending point.
    - In this case, we refer to a complete sequence of interaction, from start to finish, as an **episode**.
    - Episodic tasks come to an end whenever the agent reaches a **terminal state**.

### The Reward Hypothesis

* * *

- **Reward Hypothesis**: All goals can be framed as the maximization of (expected) cumulative reward.

### Goals and Rewards

* * *

- (Please see **Part 1** and **Part 2** to review an example of how to specify the reward signal in a real-world problem.)

### Cumulative Reward

* * *

- The **return at time step ttt** is Gt:\=Rt+1+Rt+2+Rt+3+…G\_t := R\_{t+1} + R\_{t+2} + R\_{t+3} + \\ldots Gt​:\=Rt+1​+Rt+2​+Rt+3​+…
- The agent selects actions with the goal of maximizing expected (discounted) return. (_Note: discounting is covered in the next concept._)

### Discounted Return

* * *

- The **discounted return at time step ttt** is Gt:\=Rt+1+γRt+2+γ2Rt+3+…G\_t := R\_{t+1} + \\gamma R\_{t+2} + \\gamma^2 R\_{t+3} + \\ldots Gt​:\=Rt+1​+γRt+2​+γ2Rt+3​+….
- The **discount rate γ\\gammaγ** is something that you set, to refine the goal that you have the agent.
    - It must satisfy 0≤γ≤10 \\leq \\gamma \\leq 10≤γ≤1.
    - If γ\=0\\gamma=0γ\=0, the agent only cares about the most immediate reward.
    - If γ\=1\\gamma=1γ\=1, the return is not discounted.
    - For larger values of γ\\gammaγ, the agent cares more about the distant future. Smaller values of γ\\gammaγ result in more extreme discounting, where - in the most extreme case - agent only cares about the most immediate reward.

### MDPs and One-Step Dynamics

* * *

- The **state space S\\mathcal{S}S** is the set of all (_nonterminal_) states.
- In episodic tasks, we use S+\\mathcal{S}^+S+ to refer to the set of all states, including terminal states.
- The **action space A\\mathcal{A}A** is the set of possible actions. (Alternatively, A(s)\\mathcal{A}(s)A(s) refers to the set of possible actions available in state s∈Ss \\in \\mathcal{S}s∈S.)
- (Please see **Part 2** to review how to specify the reward signal in the recycling robot example.)
- The **one-step dynamics** of the environment determine how the environment decides the state and reward at every time step. The dynamics can be defined by specifying p(s′,r∣s,a)≐P(St+1\=s′,Rt+1\=r∣St\=s,At\=a)p(s',r|s,a) \\doteq \\mathbb{P}(S\_{t+1}=s', R\_{t+1}=r|S\_{t} = s, A\_{t}=a)p(s′,r∣s,a)≐P(St+1​\=s′,Rt+1​\=r∣St​\=s,At​\=a) for each possible s′,r,s,and as', r, s, \\text{and } as′,r,s,and a.
- A **(finite) Markov Decision Process (MDP)** is defined by:
    - a (finite) set of states S\\mathcal{S}S (or S+\\mathcal{S}^+S+, in the case of an episodic task)
    - a (finite) set of actions A\\mathcal{A}A
    - a set of rewards R\\mathcal{R}R
    - the one-step dynamics of the environment
    - the discount rate γ∈\[0,1\]\\gamma \\in \[0,1\]γ∈\[0,1\]


