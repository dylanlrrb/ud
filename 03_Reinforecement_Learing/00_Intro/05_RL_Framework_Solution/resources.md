This lesson covers material in **Chapter 3** (especially 3.5-3.6) of the [textbook](http://go.udacity.com/rl-textbook).

# Bellman Equations

In this gridworld example, once the agent selects an action,

- it always moves in the chosen direction (contrasting general MDPs where the agent doesn't always have complete control over what the next state will be), and
- the reward can be predicted with complete certainty (contrasting general MDPs where the reward is a random draw from a probability distribution).

In this simple example, we saw that the value of any state can be calculated as the sum of the immediate reward and the (discounted) value of the next state.

Alexis mentioned that for a general MDP, we have to instead work in terms of an _expectation_, since it's not often the case that the immediate reward and next state can be predicted with certainty. Indeed, we saw in an earlier lesson that the reward and next state are chosen according to the one-step dynamics of the MDP. In this case, where the reward rrr and next state s′s's′ are drawn from a (conditional) probability distribution p(s′,r∣s,a)p(s',r|s,a)p(s′,r∣s,a), the **Bellman Expectation Equation (for vπv\_\\pivπ​)** expresses the value of any state sss in terms of the _expected_ immediate reward and the _expected_ value of the next state:

vπ(s)\=Eπ\[Rt+1+γvπ(St+1)∣St\=s\].v\_\\pi(s) = \\text{} \\mathbb{E}\_\\pi\[R\_{t+1} + \\gamma v\_\\pi(S\_{t+1})|S\_t =s\].vπ​(s)\=Eπ​\[Rt+1​+γvπ​(St+1​)∣St​\=s\].

## Calculating the Expectation

* * *

In the event that the agent's policy π\\piπ is **deterministic**, the agent selects action π(s)\\pi(s)π(s) when in state sss, and the Bellman Expectation Equation can be rewritten as the sum over two variables (s′s's′ and rrr):

vπ(s)\=∑s′∈S+,r∈Rp(s′,r∣s,π(s))(r+γvπ(s′))v\_\\pi(s) = \\text{} \\sum\_{s'\\in\\mathcal{S}^+, r\\in\\mathcal{R}}p(s',r|s,\\pi(s))(r+\\gamma v\_\\pi(s'))vπ​(s)\=s′∈S+,r∈R∑​p(s′,r∣s,π(s))(r+γvπ​(s′))

In this case, we multiply the sum of the reward and discounted value of the next state (r+γvπ(s′))(r+\\gamma v\_\\pi(s'))(r+γvπ​(s′)) by its corresponding probability p(s′,r∣s,π(s))p(s',r|s,\\pi(s))p(s′,r∣s,π(s)) and sum over all possibilities to yield the expected value.

If the agent's policy π\\piπ is **stochastic**, the agent selects action aaa with probability π(a∣s)\\pi(a|s)π(a∣s) when in state sss, and the Bellman Expectation Equation can be rewritten as the sum over three variables (s′s's′, rrr, and aaa):

vπ(s)\=∑s′∈S+,r∈R,a∈A(s)π(a∣s)p(s′,r∣s,a)(r+γvπ(s′))v\_\\pi(s) = \\text{} \\sum\_{s'\\in\\mathcal{S}^+, r\\in\\mathcal{R},a\\in\\mathcal{A}(s)}\\pi(a|s)p(s',r|s,a)(r+\\gamma v\_\\pi(s'))vπ​(s)\=s′∈S+,r∈R,a∈A(s)∑​π(a∣s)p(s′,r∣s,a)(r+γvπ​(s′))

In this case, we multiply the sum of the reward and discounted value of the next state (r+γvπ(s′))(r+\\gamma v\_\\pi(s'))(r+γvπ​(s′)) by its corresponding probability π(a∣s)p(s′,r∣s,a)\\pi(a|s)p(s',r|s,a)π(a∣s)p(s′,r∣s,a) and sum over all possibilities to yield the expected value.

## There are 3 more Bellman Equations!

* * *

In this video, you learned about one Bellman equation, but there are 3 more, for a total of 4 Bellman equations.

> All of the Bellman equations attest to the fact that _value functions satisfy recursive relationships_.

For instance, the **Bellman Expectation Equation (for vπv\_\\pivπ​)** shows that it is possible to relate the value of a state to the values of all of its possible successor states.

After finishing this lesson, you are encouraged to read about the remaining three Bellman equations in sections 3.5 and 3.6 of the [textbook](http://go.udacity.com/rl-textbook). The Bellman equations are incredibly useful to the theory of MDPs.


# Quiz: State-Value Functions

In this quiz, you will calculate the value function corresponding to a particular policy.

Each of the nine states in the MDP is labeled as one of S+\={s1,s2,…,s9}\\mathcal{S}^+ = \\{s\_1, s\_2, \\ldots, s\_9 \\} S+\={s1​,s2​,…,s9​}, where s9s\_9s9​ is a terminal state.

Consider the (deterministic) policy that is indicated (in orange) in the figure below.

![](https://video.udacity-data.com/topher/2017/September/59c823a0_screen-shot-2017-09-24-at-4.28.04-pm/screen-shot-2017-09-24-at-4.28.04-pm.png)

The policy π\\piπ is given by:

> π(s1)\=right\\pi(s\_1) = \\text{right}π(s1​)\=right

> π(s2)\=right\\pi(s\_2) = \\text{right}π(s2​)\=right

> π(s3)\=down\\pi(s\_3) = \\text{down}π(s3​)\=down

> π(s4)\=up\\pi(s\_4) = \\text{up}π(s4​)\=up

> π(s5)\=right\\pi(s\_5) = \\text{right}π(s5​)\=right

> π(s6)\=down\\pi(s\_6) = \\text{down}π(s6​)\=down

> π(s7)\=right\\pi(s\_7) = \\text{right}π(s7​)\=right

> π(s8)\=right\\pi(s\_8) = \\text{right}π(s8​)\=right

Recall that since s9s\_9s9​ is a terminal state, the episode ends immediately if the agent begins in this state. So, the agent will not have to choose an action (so, we won't include s9s\_9s9​ in the domain of the policy), and vπ(s9)\=0v\_\\pi(s\_9) = 0vπ​(s9​)\=0.

Take the time now to calculate the state-value function vπv\_\\pivπ​ that corresponds to the policy. (_You may find that the Bellman expectation equation saves you a lot of work!_)

**Assume γ\=1\\gamma = 1γ\=1.**


# Quiz: Optimal Policies

If the state space S\\mathcal{S}S and action space A\\mathcal{A}A are finite, we can represent the optimal action-value function q∗q\_\*q∗​ in a table, where we have one entry for each possible environment state s∈Ss \\in \\mathcal{S}s∈S and action a∈Aa\\in\\mathcal{A}a∈A.

The value for a particular state-action pair s,as,as,a is the expected return if the agent starts in state sss, takes action aaa, and then henceforth follows the optimal policy π∗\\pi\_\*π∗​.

We have populated some values for a hypothetical Markov decision process (MDP) (where S\={s1,s2,s3}\\mathcal{S}=\\{ s\_1, s\_2, s\_3 \\}S\={s1​,s2​,s3​} and A\={a1,a2,a3}\\mathcal{A}=\\{a\_1, a\_2, a\_3\\}A\={a1​,a2​,a3​}) below.

![](https://video.udacity-data.com/topher/2017/September/59c98891_screen-shot-2017-09-25-at-5.51.40-pm/screen-shot-2017-09-25-at-5.51.40-pm.png)

You learned in the previous concept that once the agent has determined the optimal action-value function q∗q\_\*q∗​, it can quickly obtain an optimal policy π∗\\pi\_\*π∗​ by setting π∗(s)\=argmaxa∈A(s)q∗(s,a)\\pi\_\*(s) = \\arg\\max\_{a\\in\\mathcal{A}(s)} q\_\*(s,a)π∗​(s)\=argmaxa∈A(s)​q∗​(s,a) for all s∈Ss\\in\\mathcal{S}s∈S.

To see _why_ this should be the case, note that it must hold that v∗(s)\=maxa∈A(s)q∗(s,a)v\_\*(s) = \\max\_{a\\in\\mathcal{A}(s)} q\_\*(s,a)v∗​(s)\=maxa∈A(s)​q∗​(s,a).

In the event that there is some state s∈Ss\\in\\mathcal{S}s∈S for which multiple actions a∈A(s)a\\in\\mathcal{A}(s)a∈A(s) maximize the optimal action-value function, you can construct an optimal policy by placing any amount of probability on any of the (maximizing) actions. You need only ensure that the actions that do not maximize the action-value function (for a particular state) are given 0% probability under the policy.

Towards constructing the optimal policy, we can begin by selecting the entries that maximize the action-value function, for each row (or state).

![](https://video.udacity-data.com/topher/2017/September/59c98bf4_screen-shot-2017-09-25-at-6.02.37-pm/screen-shot-2017-09-25-at-6.02.37-pm.png)

Thus, the optimal policy π∗\\pi\_\*π∗​ for the corresponding MDP must satisfy:

- π∗(s1)\=a2\\pi\_\*(s\_1) = a\_2π∗​(s1​)\=a2​ (or, equivalently, π∗(a2∣s1)\=1\\pi\_\*(a\_2| s\_1) = 1π∗​(a2​∣s1​)\=1), and
- π∗(s2)\=a3\\pi\_\*(s\_2) = a\_3π∗​(s2​)\=a3​ (or, equivalently, π∗(a3∣s2)\=1\\pi\_\*(a\_3| s\_2) = 1π∗​(a3​∣s2​)\=1).

This is because a2\=argmaxa∈A(s1)q∗(s1,a)a\_2 = \\arg\\max\_{a\\in\\mathcal{A}(s\_1)}q\_\*(s\_1,a)a2​\=argmaxa∈A(s1​)​q∗​(s1​,a), and a3\=argmaxa∈A(s2)q∗(s2,a)a\_3 = \\arg\\max\_{a\\in\\mathcal{A}(s\_2)}q\_\*(s\_2,a)a3​\=argmaxa∈A(s2​)​q∗​(s2​,a).

In other words, under the optimal policy, the agent must choose action a2a\_2a2​ when in state s1s\_1s1​, and it will choose action a3a\_3a3​ when in state s2s\_2s2​.

As for state s3s\_3s3​, note that a1,a2∈argmaxa∈A(s3)q∗(s3,a)a\_1, a\_2 \\in \\arg\\max\_{a\\in\\mathcal{A}(s\_3)}q\_\*(s\_3,a)a1​,a2​∈argmaxa∈A(s3​)​q∗​(s3​,a). Thus, the agent can choose either action a1a\_1a1​ or a2a\_2a2​ under the optimal policy, but it can never choose action a3a\_3a3​. That is, the optimal policy π∗\\pi\_\*π∗​ must satisfy:

- π∗(a1∣s3)\=p\\pi\_\*(a\_1| s\_3) = pπ∗​(a1​∣s3​)\=p,
- π∗(a2∣s3)\=q\\pi\_\*(a\_2| s\_3) = qπ∗​(a2​∣s3​)\=q, and
- π∗(a3∣s3)\=0\\pi\_\*(a\_3| s\_3) = 0π∗​(a3​∣s3​)\=0,

where p,q≥0p,q\\geq 0p,q≥0, and p+q\=1p + q = 1p+q\=1.

## Question

Consider a different MDP, with a different corresponding optimal action-value function. Please use this action-value function to answer the following question.

![](https://video.udacity-data.com/topher/2017/September/59c9b917_screen-shot-2017-09-25-at-9.18.00-pm/screen-shot-2017-09-25-at-9.18.00-pm.png)


# Summary

![](https://video.udacity-data.com/topher/2017/September/59c93080_screen-shot-2017-09-25-at-11.35.38-am/screen-shot-2017-09-25-at-11.35.38-am.png)

State-value function for golf-playing agent (Sutton and Barto, 2017)

### Policies

* * *

- A **deterministic policy** is a mapping π:S→A\\pi: \\mathcal{S}\\to\\mathcal{A}π:S→A. For each state s∈Ss\\in\\mathcal{S}s∈S, it yields the action a∈Aa\\in\\mathcal{A}a∈A that the agent will choose while in state sss.
- A **stochastic policy** is a mapping π:S×A→\[0,1\]\\pi: \\mathcal{S}\\times\\mathcal{A}\\to \[0,1\]π:S×A→\[0,1\]. For each state s∈Ss\\in\\mathcal{S}s∈S and action a∈Aa\\in\\mathcal{A}a∈A, it yields the probability π(a∣s)\\pi(a|s)π(a∣s) that the agent chooses action aaa while in state sss.

### State-Value Functions

* * *

- The **state-value function** for a policy π\\piπ is denoted vπv\_\\pivπ​. For each state s∈Ss \\in\\mathcal{S}s∈S, it yields the expected return if the agent starts in state sss and then uses the policy to choose its actions for all time steps. That is, vπ(s)≐Eπ\[Gt∣St\=s\]v\_\\pi(s) \\doteq \\text{} \\mathbb{E}\_\\pi\[G\_t|S\_t=s\]vπ​(s)≐Eπ​\[Gt​∣St​\=s\]. We refer to vπ(s)v\_\\pi(s)vπ​(s) as the **value of state sss under policy π\\piπ**.
- The notation Eπ\[⋅\]\\mathbb{E}\_\\pi\[\\cdot\]Eπ​\[⋅\] is borrowed from the suggested textbook, where Eπ\[⋅\]\\mathbb{E}\_\\pi\[\\cdot\]Eπ​\[⋅\] is defined as the expected value of a random variable, given that the agent follows policy π\\piπ.

### Bellman Equations

* * *

- The **Bellman expectation equation for vπv\_\\pivπ​** is: vπ(s)\=Eπ\[Rt+1+γvπ(St+1)∣St\=s\].v\_\\pi(s) = \\text{} \\mathbb{E}\_\\pi\[R\_{t+1} + \\gamma v\_\\pi(S\_{t+1})|S\_t =s\].vπ​(s)\=Eπ​\[Rt+1​+γvπ​(St+1​)∣St​\=s\].

### Optimality

* * *

- A policy π′\\pi'π′ is defined to be better than or equal to a policy π\\piπ if and only if vπ′(s)≥vπ(s)v\_{\\pi'}(s) \\geq v\_\\pi(s)vπ′​(s)≥vπ​(s) for all s∈Ss\\in\\mathcal{S}s∈S.
- An **optimal policy π∗\\pi\_\*π∗​** satisfies π∗≥π\\pi\_\* \\geq \\piπ∗​≥π for all policies π\\piπ. An optimal policy is guaranteed to exist but may not be unique.
- All optimal policies have the same state-value function v∗v\_\*v∗​, called the **optimal state-value function**.

### Action-Value Functions

* * *

- The **action-value function** for a policy π\\piπ is denoted qπq\_\\piqπ​. For each state s∈Ss \\in\\mathcal{S}s∈S and action a∈Aa \\in\\mathcal{A}a∈A, it yields the expected return if the agent starts in state sss, takes action aaa, and then follows the policy for all future time steps. That is, qπ(s,a)≐Eπ\[Gt∣St\=s,At\=a\]q\_\\pi(s,a) \\doteq \\mathbb{E}\_\\pi\[G\_t|S\_t=s, A\_t=a\]qπ​(s,a)≐Eπ​\[Gt​∣St​\=s,At​\=a\]. We refer to qπ(s,a)q\_\\pi(s,a)qπ​(s,a) as the **value of taking action aaa in state sss under a policy π\\piπ** (or alternatively as the **value of the state-action pair s,as, as,a**).
- All optimal policies have the same action-value function q∗q\_\*q∗​, called the **optimal action-value function**.

### Optimal Policies

* * *

- Once the agent determines the optimal action-value function q∗q\_\*q∗​, it can quickly obtain an optimal policy π∗\\pi\_\*π∗​ by setting π∗(s)\=argmaxa∈A(s)q∗(s,a)\\pi\_\*(s) = \\arg\\max\_{a\\in\\mathcal{A}(s)} q\_\*(s,a)π∗​(s)\=argmaxa∈A(s)​q∗​(s,a).
