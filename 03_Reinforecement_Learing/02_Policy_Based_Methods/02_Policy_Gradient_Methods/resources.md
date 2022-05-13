# Connections to Supervised Learning

Policy gradient methods are very similar to supervised learning.

## (Optional) Learn More

* * *

To further explore the connections between policy gradient methods and supervised learning, you're encouraged to check out Andrej Karpathy's [famous blog post](http://karpathy.github.io/2016/05/31/rl/).

![Learn more about the connections between supervised learning and reinforcement learning.](https://video.udacity-data.com/topher/2018/July/5b5b71dd_screen-shot-2018-07-27-at-2.25.43-pm/screen-shot-2018-07-27-at-2.25.43-pm.png)

Learn more about the connections between supervised learning and reinforcement learning. ([Source](http://karpathy.github.io/2016/05/31/rl/))


# Problem Setup

We're now ready to get started with rigorously defining how policy gradient methods will work.

## Important Note

* * *

Before moving on, make sure it's clear to you that the equation discussed in the video (and shown below) calculates an [expectation](https://en.wikipedia.org/wiki/Expected_value).

U(θ)\=∑τP(τ;θ)R(τ)U(\\theta) = \\sum\_\\tau \\mathbb{P}(\\tau;\\theta)R(\\tau)U(θ)\=τ∑​P(τ;θ)R(τ)

To see how it corresponds to the **expected return**, note that we've expressed the **return** R(τ)R(\\tau)R(τ) as a function of the trajectory τ\\tauτ. Then, we calculate the weighted average (_where the weights are given by P(τ;θ)\\mathbb{P}(\\tau;\\theta)P(τ;θ)_) of all possible values that the return R(τ)R(\\tau)R(τ) can take.

## Why Trajectories?

* * *

You may be wondering: _why are we using trajectories instead of episodes?_ The answer is that maximizing expected return over trajectories (instead of episodes) lets us search for optimal policies for both episodic _and continuing_ tasks!

That said, for many episodic tasks, it often makes sense to just use the full episode. In particular, for the case of the video game example described in the lessons, reward is only delivered at the end of the episode. In this case, in order to estimate the expected return, the trajectory should correspond to the full episode; otherwise, we don't have enough reward information to meaningfully estimate the expected return.


# REINFORCE

You've learned that our goal is to find the values of the weights θ\\thetaθ in the neural network that maximize the expected return UUU

U(θ)\=∑τP(τ;θ)R(τ)U(\\theta) = \\sum\_\\tau P(\\tau;\\theta)R(\\tau)U(θ)\=τ∑​P(τ;θ)R(τ)

where τ\\tauτ is an arbitrary trajectory. One way to determine the value of θ\\thetaθ that maximizes this function is through **gradient ascent**. This algorithm is closely related to **gradient descent**, where the differences are that:

- gradient descent is designed to find the **minimum** of a function, whereas gradient ascent will find the **maximum**, and
- gradient descent steps in the direction of the **negative gradient**, whereas gradient ascent steps in the direction of the **gradient**.

Our update step for gradient ascent appears as follows:

θ←θ+α∇θU(θ)\\theta \\leftarrow \\theta + \\alpha \\nabla\_\\theta U(\\theta)θ←θ+α∇θ​U(θ)

where α\\alphaα is the step size that is generally allowed to decay over time. Once we know how to calculate or estimate this gradient, we can repeatedly apply this update step, in the hopes that θ\\thetaθ converges to the value that maximizes U(θ)U(\\theta)U(θ).

## Video

* * *

## Pseudocode

* * *

The algorithm described in the video is known as **REINFORCE**. The pseudocode is summarized below.

1. Use the policy πθ\\pi\_\\thetaπθ​ to collect mmm trajectories {τ(1),τ(2),…,τ(m)}\\{ \\tau^{(1)}, \\tau^{(2)}, \\ldots, \\tau^{(m)}\\}{τ(1),τ(2),…,τ(m)} with horizon HHH. We refer to the iii\-th trajectory as τ(i)\=(s0(i),a0(i),…,sH(i),aH(i),sH+1(i))\\tau^{(i)} = (s\_0^{(i)}, a\_0^{(i)}, \\ldots, s\_H^{(i)}, a\_H^{(i)}, s\_{H+1}^{(i)})τ(i)\=(s0(i)​,a0(i)​,…,sH(i)​,aH(i)​,sH+1(i)​).
2. Use the trajectories to estimate the gradient ∇θU(θ)\\nabla\_\\theta U(\\theta)∇θ​U(θ): ∇θU(θ)≈g^:\=1m∑i\=1m∑t\=0H∇θlogπθ(at(i)∣st(i))R(τ(i))\\nabla\_\\theta U(\\theta) \\approx \\hat{g} := \\frac{1}{m}\\sum\_{i=1}^m \\sum\_{t=0}^{H} \\nabla\_\\theta \\log \\pi\_\\theta(a\_t^{(i)}|s\_t^{(i)}) R(\\tau^{(i)})∇θ​U(θ)≈g^​:\=m1​i\=1∑m​t\=0∑H​∇θ​logπθ​(at(i)​∣st(i)​)R(τ(i))
3. Update the weights of the policy: θ←θ+αg^\\theta \\leftarrow \\theta + \\alpha \\hat{g}θ←θ+αg^​
4. Loop over steps 1-3.



![Behavior of different optimizers for stochastic gradient descent. ](https://video.udacity-data.com/topher/2018/July/5b59f39e_grad-descent/grad-descent.gif)

Behavior of different optimizers for stochastic gradient descent. ([Source](http://ruder.io/optimizing-gradient-descent/))

# (Optional) Derivation

If you'd like to learn how to derive the equation that we use to approximate the gradient, please read the text below. Specifically, you'll learn how to derive

∇θU(θ)≈g^\=1m∑i\=1m∑t\=0H∇θlogπθ(at(i)∣st(i))R(τ(i))\\nabla\_\\theta U(\\theta) \\approx \\hat{g}= \\frac{1}{m}\\sum\_{i=1}^m \\sum\_{t=0}^{H} \\nabla\_\\theta \\log \\pi\_\\theta(a\_t^{(i)}|s\_t^{(i)}) R(\\tau^{(i)})∇θ​U(θ)≈g^​\=m1​i\=1∑m​t\=0∑H​∇θ​logπθ​(at(i)​∣st(i)​)R(τ(i))

This derivation is **optional** and can be safely skipped.

## Likelihood Ratio Policy Gradient

* * *

We'll begin by exploring how to calculate the gradient ∇θU(θ)\\nabla\_\\theta U(\\theta)∇θ​U(θ). The calculation proceeds as follows:

∇θU(θ)\=∇θ∑τP(τ;θ)R(τ)(1)\=∑τ∇θP(τ;θ)R(τ)(2)\=∑τP(τ;θ)P(τ;θ)∇θP(τ;θ)R(τ)(3)\=∑τP(τ;θ)∇θP(τ;θ)P(τ;θ)R(τ)(4)\=∑τP(τ;θ)∇θlogP(τ;θ)R(τ)(5)\\begin{aligned}\\nabla\_\\theta U(\\theta) &= \\nabla\_\\theta \\sum\_\\tau P(\\tau;\\theta)R(\\tau) & (1)\\\\ &= \\sum\_\\tau \\nabla\_\\theta P(\\tau;\\theta)R(\\tau) & (2)\\\\ &= \\sum\_\\tau \\frac{P(\\tau;\\theta)}{P(\\tau;\\theta)} \\nabla\_\\theta P(\\tau;\\theta)R(\\tau) & (3)\\\\ &= \\sum\_\\tau P(\\tau;\\theta) \\frac{\\nabla\_\\theta P(\\tau;\\theta)}{P(\\tau;\\theta)}R(\\tau) & (4)\\\\ &= \\sum\_\\tau P(\\tau;\\theta) \\nabla\_\\theta \\log P(\\tau;\\theta) R(\\tau) & (5) \\end{aligned}∇θ​U(θ)​\=∇θ​τ∑​P(τ;θ)R(τ)\=τ∑​∇θ​P(τ;θ)R(τ)\=τ∑​P(τ;θ)P(τ;θ)​∇θ​P(τ;θ)R(τ)\=τ∑​P(τ;θ)P(τ;θ)∇θ​P(τ;θ)​R(τ)\=τ∑​P(τ;θ)∇θ​logP(τ;θ)R(τ)​(1)(2)(3)(4)(5)​

First, we note line (1) follows directly from U(θ)\=∑τP(τ;θ)R(τ)U(\\theta) = \\sum\_\\tau P(\\tau;\\theta)R(\\tau)U(θ)\=∑τ​P(τ;θ)R(τ), where we've only taken the gradient of both sides.

Then, we can get line (2) by just noticing that we can rewrite the gradient of the sum as the sum of the gradients.

In line (3), we only multiply every term in the sum by P(τ;θ)P(τ;θ)\\frac{P(\\tau;\\theta)}{P(\\tau;\\theta)}P(τ;θ)P(τ;θ)​, which is perfectly allowed because this fraction is equal to one!

Next, line (4) is just a simple rearrangement of the terms from the previous line. That is, P(τ;θ)P(τ;θ)∇θP(τ;θ)\=P(τ;θ)∇θP(τ;θ)P(τ;θ)\\frac{P(\\tau;\\theta)}{P(\\tau;\\theta)} \\nabla\_\\theta P(\\tau;\\theta) = P(\\tau;\\theta) \\frac{\\nabla\_\\theta P(\\tau;\\theta)}{P(\\tau;\\theta)}P(τ;θ)P(τ;θ)​∇θ​P(τ;θ)\=P(τ;θ)P(τ;θ)∇θ​P(τ;θ)​.

Finally, line (5) follows from the chain rule, and the fact that the gradient of the log of a function is always equal to the gradient of the function, divided by the function. (_In case it helps to see this with simpler notation, recall that ∇xlogf(x)\=∇xf(x)f(x)\\nabla\_x \\log f(x) = \\frac{\\nabla\_x f(x)}{f(x)}∇x​logf(x)\=f(x)∇x​f(x)​._) Thus, ∇θlogP(τ;θ)\=∇θP(τ;θ)P(τ;θ)\\nabla\_\\theta \\log P(\\tau;\\theta) = \\frac{\\nabla\_\\theta P(\\tau;\\theta)}{P(\\tau;\\theta)}∇θ​logP(τ;θ)\=P(τ;θ)∇θ​P(τ;θ)​.

The final "trick" that yields line (5) (i.e., ∇θlogP(τ;θ)\=∇θP(τ;θ)P(τ;θ)\\nabla\_\\theta \\log P(\\tau;\\theta) = \\frac{\\nabla\_\\theta P(\\tau;\\theta)}{P(\\tau;\\theta)}∇θ​logP(τ;θ)\=P(τ;θ)∇θ​P(τ;θ)​) is referred to as the **likelihood ratio trick** or **REINFORCE trick**.

Likewise, it is common to refer to the gradient as the **likelihood ratio policy gradient**: ∇θU(θ)\=∑τP(τ;θ)∇θlogP(τ;θ)R(τ)\\nabla\_\\theta U(\\theta) = \\sum\_\\tau P(\\tau;\\theta) \\nabla\_\\theta \\log P(\\tau;\\theta) R(\\tau)∇θ​U(θ)\=τ∑​P(τ;θ)∇θ​logP(τ;θ)R(τ)

Once we’ve written the gradient as an expected value in this way, it becomes much easier to estimate.

## Sample-Based Estimate

* * *

In the video on the previous page, you learned that we can approximate the likelihood ratio policy gradient with a sample-based average, as shown below:

∇θU(θ)≈1m∑i\=1m∇θlogP(τ(i);θ)R(τ(i))\\nabla\_\\theta U(\\theta) \\approx \\frac{1}{m}\\sum\_{i=1}^m \\nabla\_\\theta \\log \\mathbb{P}(\\tau^{(i)};\\theta)R(\\tau^{(i)})∇θ​U(θ)≈m1​i\=1∑m​∇θ​logP(τ(i);θ)R(τ(i))

where each τ(i)\\tau^{(i)}τ(i) is a sampled trajectory.

## Finishing the Calculation

* * *

Before calculating the expression above, we will need to further simplify ∇θlogP(τ(i);θ)\\nabla\_\\theta \\log \\mathbb{P}(\\tau^{(i)};\\theta)∇θ​logP(τ(i);θ). The derivation proceeds as follows:

∇θlogP(τ(i);θ)\=∇θlog\[∏t\=0HP(st+1(i)∣st(i),at(i))πθ(at(i)∣st(i))\](1)\=∇θ\[∑t\=0HlogP(st+1(i)∣st(i),at(i))+∑t\=0Hlogπθ(at(i)∣st(i))\](2)\=∇θ∑t\=0HlogP(st+1(i)∣st(i),at(i))+∇θ∑t\=0Hlogπθ(at(i)∣st(i))(3)\=∇θ∑t\=0Hlogπθ(at(i)∣st(i))(4)\=∑t\=0H∇θlogπθ(at(i)∣st(i))(5)\\begin{aligned} \\nabla\_\\theta \\log \\mathbb{P}(\\tau^{(i)};\\theta) &= \\nabla\_\\theta \\log \\Bigg\[ \\prod\_{t=0}^{H} \\mathbb{P}(s\_{t+1}^{(i)}|s\_{t}^{(i)}, a\_t^{(i)} )\\pi\_\\theta(a\_t^{(i)}|s\_t^{(i)}) \\Bigg\] & (1)\\\\ &= \\nabla\_\\theta \\Bigg\[ \\sum\_{t=0}^{H} \\log \\mathbb{P}(s\_{t+1}^{(i)}|s\_{t}^{(i)}, a\_t^{(i)} ) + \\sum\_{t=0}^{H}\\log \\pi\_\\theta(a\_t^{(i)}|s\_t^{(i)}) \\Bigg\] & (2)\\\\ &= \\nabla\_\\theta\\sum\_{t=0}^{H} \\log \\mathbb{P}(s\_{t+1}^{(i)}|s\_{t}^{(i)}, a\_t^{(i)} ) + \\nabla\_\\theta \\sum\_{t=0}^{H}\\log \\pi\_\\theta(a\_t^{(i)}|s\_t^{(i)}) & (3)\\\\ &= \\nabla\_\\theta \\sum\_{t=0}^{H}\\log \\pi\_\\theta(a\_t^{(i)}|s\_t^{(i)}) & (4)\\\\ &= \\sum\_{t=0}^{H} \\nabla\_\\theta \\log \\pi\_\\theta(a\_t^{(i)}|s\_t^{(i)}) & (5) \\end{aligned}∇θ​logP(τ(i);θ)​\=∇θ​log\[t\=0∏H​P(st+1(i)​∣st(i)​,at(i)​)πθ​(at(i)​∣st(i)​)\]\=∇θ​\[t\=0∑H​logP(st+1(i)​∣st(i)​,at(i)​)+t\=0∑H​logπθ​(at(i)​∣st(i)​)\]\=∇θ​t\=0∑H​logP(st+1(i)​∣st(i)​,at(i)​)+∇θ​t\=0∑H​logπθ​(at(i)​∣st(i)​)\=∇θ​t\=0∑H​logπθ​(at(i)​∣st(i)​)\=t\=0∑H​∇θ​logπθ​(at(i)​∣st(i)​)​(1)(2)(3)(4)(5)​

First, line (1) shows how to calculate the probability of an arbitrary trajectory τ(i)\\tau^{(i)}τ(i). Namely, P(τ(i);θ)\=∏t\=0HP(st+1(i)∣st(i),at(i))πθ(at(i)∣st(i))\\mathbb{P}(\\tau^{(i)};\\theta) = \\prod\_{t=0}^{H} \\mathbb{P}(s\_{t+1}^{(i)}|s\_{t}^{(i)}, a\_t^{(i)} )\\pi\_\\theta(a\_t^{(i)}|s\_t^{(i)}) P(τ(i);θ)\=∏t\=0H​P(st+1(i)​∣st(i)​,at(i)​)πθ​(at(i)​∣st(i)​), where we have to take into account the action-selection probabilities from the policy and the state transition dynamics of the MDP.

Then, line (2) follows from the fact that the log of a product is equal to the sum of the logs.

Then, line (3) follows because the gradient of the sum can be written as the sum of gradients.

Next, line (4) holds, because ∑t\=0HlogP(st+1(i)∣st(i),at(i))\\sum\_{t=0}^{H} \\log \\mathbb{P}(s\_{t+1}^{(i)}|s\_{t}^{(i)}, a\_t^{(i)} )∑t\=0H​logP(st+1(i)​∣st(i)​,at(i)​) has no dependence on θ\\thetaθ, so ∇θ∑t\=0HlogP(st+1(i)∣st(i),at(i))\=0\\nabla\_\\theta\\sum\_{t=0}^{H} \\log \\mathbb{P}(s\_{t+1}^{(i)}|s\_{t}^{(i)}, a\_t^{(i)} )=0∇θ​∑t\=0H​logP(st+1(i)​∣st(i)​,at(i)​)\=0.

Finally, line (5) holds, because we can rewrite the gradient of the sum as the sum of gradients.

## That's it!

* * *

Plugging in the calculation above yields the equation for estimating the gradient: ∇θU(θ)≈g^\=1m∑i\=1m∑t\=0H∇θlogπθ(at(i)∣st(i))R(τ(i))\\nabla\_\\theta U(\\theta) \\approx \\hat{g} = \\frac{1}{m}\\sum\_{i=1}^m \\sum\_{t=0}^{H} \\nabla\_\\theta \\log \\pi\_\\theta(a\_t^{(i)}|s\_t^{(i)}) R(\\tau^{(i)})∇θ​U(θ)≈g^​\=m1​i\=1∑m​t\=0∑H​∇θ​logπθ​(at(i)​∣st(i)​)R(τ(i))



# What's Next?

In this lesson, you've learned all about the REINFORCE algorithm, which was illustrated with a toy environment with a **_discrete_** action space. But it's also important to mention that REINFORCE can also be used to solve environments with continuous action spaces!

For an environment with a continuous action space, the corresponding policy network could have an output layer that parametrizes a [continuous probability distribution](https://en.wikipedia.org/wiki/Probability_distribution#Continuous_probability_distribution).

For instance, assume the output layer returns the mean μ\\muμ and variance σ2\\sigma^2σ2 of a [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution).

![Probability density function corresponding to normal distribution (Source: Wikipedia)](https://video.udacity-data.com/topher/2018/August/5b81ba57_350px-normal-distribution-pdf/350px-normal-distribution-pdf.png)

Probability density function corresponding to normal distribution (Source: Wikipedia)

Then in order to select an action, the agent needs only to pass the most recent state sts\_tst​ as input to the network, and then use the output mean μ\\muμ and variance σ2\\sigma^2σ2 to sample from the distribution at∼N(μ,σ2)a\_t\\sim\\mathcal{N}(\\mu, \\sigma^2)at​∼N(μ,σ2).

This should work in theory, but it's unlikely to perform well in practice! To improve performance with continuous action spaces, we'll have to make some small modifications to the REINFORCE algorithm, and you'll learn more about these modifications in the upcoming lessons.



# Summary

![REINFORCE increases the probability of "good" actions and decreases the probability of "bad" actions.](https://video.udacity-data.com/topher/2018/July/5b4e634b_screen-shot-2018-07-17-at-4.44.10-pm/screen-shot-2018-07-17-at-4.44.10-pm.png)

REINFORCE increases the probability of "good" actions and decreases the probability of "bad" actions. ([Source](https://blog.openai.com/evolution-strategies/))

### What are Policy Gradient Methods?

* * *

- **Policy-based methods** are a class of algorithms that search directly for the optimal policy, without simultaneously maintaining value function estimates.
- **Policy gradient methods** are a subclass of policy-based methods that estimate the weights of an optimal policy through gradient ascent.
- In this lesson, we represent the policy with a neural network, where our goal is to find the weights θ\\thetaθ of the network that maximize expected return.

### The Big Picture

* * *

- The policy gradient method will iteratively amend the policy network weights to:
    - make (state, action) pairs that resulted in positive return more likely, and
    - make (state, action) pairs that resulted in negative return less likely.

### Problem Setup

* * *

- A **trajectory** τ\\tauτ is a state-action sequence s0,a0,…,sH,aH,sH+1s\_0, a\_0, \\ldots, s\_H, a\_H, s\_{H+1}s0​,a0​,…,sH​,aH​,sH+1​.
- In this lesson, we will use the notation R(τ)R(\\tau)R(τ) to refer to the return corresponding to trajectory τ\\tauτ.
- Our goal is to find the weights θ\\thetaθ of the policy network to maximize the **expected return** U(θ):\=∑τP(τ;θ)R(τ)U(\\theta) := \\sum\_\\tau \\mathbb{P}(\\tau;\\theta)R(\\tau)U(θ):\=∑τ​P(τ;θ)R(τ).

### REINFORCE

* * *

- The pseudocode for REINFORCE is as follows:
    1. Use the policy πθ\\pi\_\\thetaπθ​ to collect mmm trajectories {τ(1),τ(2),…,τ(m)}\\{ \\tau^{(1)}, \\tau^{(2)}, \\ldots, \\tau^{(m)}\\}{τ(1),τ(2),…,τ(m)} with horizon HHH. We refer to the iii\-th trajectory as τ(i)\=(s0(i),a0(i),…,sH(i),aH(i),sH+1(i))\\tau^{(i)} = (s\_0^{(i)}, a\_0^{(i)}, \\ldots, s\_H^{(i)}, a\_H^{(i)}, s\_{H+1}^{(i)})τ(i)\=(s0(i)​,a0(i)​,…,sH(i)​,aH(i)​,sH+1(i)​).
    2. Use the trajectories to estimate the gradient ∇θU(θ)\\nabla\_\\theta U(\\theta)∇θ​U(θ): ∇θU(θ)≈g^:\=1m∑i\=1m∑t\=0H∇θlogπθ(at(i)∣st(i))R(τ(i))\\nabla\_\\theta U(\\theta) \\approx \\hat{g} := \\frac{1}{m}\\sum\_{i=1}^m \\sum\_{t=0}^{H} \\nabla\_\\theta \\log \\pi\_\\theta(a\_t^{(i)}|s\_t^{(i)}) R(\\tau^{(i)})∇θ​U(θ)≈g^​:\=m1​i\=1∑m​t\=0∑H​∇θ​logπθ​(at(i)​∣st(i)​)R(τ(i))
    3. Update the weights of the policy: θ←θ+αg^\\theta \\leftarrow \\theta + \\alpha \\hat{g}θ←θ+αg^​
    4. Loop over steps 1-3.

### Derivation

* * *

- We derived the **likelihood ratio policy gradient**: ∇θU(θ)\=∑τP(τ;θ)∇θlogP(τ;θ)R(τ)\\nabla\_\\theta U(\\theta) = \\sum\_\\tau \\mathbb{P}(\\tau;\\theta)\\nabla\_\\theta \\log \\mathbb{P}(\\tau;\\theta)R(\\tau) ∇θ​U(θ)\=∑τ​P(τ;θ)∇θ​logP(τ;θ)R(τ).
- We can approximate the gradient above with a sample-weighted average: ∇θU(θ)≈1m∑i\=1m∇θlogP(τ(i);θ)R(τ(i))\\nabla\_\\theta U(\\theta) \\approx \\frac{1}{m}\\sum\_{i=1}^m \\nabla\_\\theta \\log \\mathbb{P}(\\tau^{(i)};\\theta)R(\\tau^{(i)}) ∇θ​U(θ)≈m1​i\=1∑m​∇θ​logP(τ(i);θ)R(τ(i)).
- We calculated the following: ∇θlogP(τ(i);θ)\=∑t\=0H∇θlogπθ(at(i)∣st(i))\\nabla\_\\theta \\log \\mathbb{P}(\\tau^{(i)};\\theta) = \\sum\_{t=0}^{H} \\nabla\_\\theta \\log \\pi\_\\theta (a\_t^{(i)}|s\_t^{(i)}) ∇θ​logP(τ(i);θ)\=t\=0∑H​∇θ​logπθ​(at(i)​∣st(i)​).

### What's Next?

* * *

- REINFORCE can solve Markov Decision Processes (MDPs) with either discrete or continuous action spaces.



