# Lesson Preview

State-of-the-art RL algorithms contain many important tweaks in addition to simple value-based or policy-based methods. One of these key improvements is called Proximal Policy Optimization (PPO) -- also closely related to Trust Region Policy Optimization (TRPO). It has allowed faster and more stable learning. From developing agile robots, to creating expert level gaming AI, PPO has proven useful in a wide domain of applications, and has become part of the standard toolkits in complicated learning environments.

In this lesson, we will first review the most basic policy gradient algorithm -- REINFORCE, and discuss issues associated with the algorithm. We will get an in-depth understanding of why these problems arise, and find ways to fix them. The solutions will lead us to PPO. Our lesson will focus on learning the intuitions behind why and how PPO improves learning, and implement it to teach a computer to play Atari-Pong, using only the pixels as input (see video below). Let's dive in!

_The idea of PPO was published by the team at OpenAI, and you can read their paper through this [link](https://arxiv.org/abs/1707.06347)_


# Beyond REINFORCE

Here, we briefly review key ingredients of the REINFORCE algorithm.

REINFORCE works as follows: First, we initialize a random policy πθ(a;s)\\pi\_\\theta(a;s)πθ​(a;s), and using the policy we collect a trajectory -- or a list of (state, actions, rewards) at each time step:

s1,a1,r1,s2,a2,r2,... s\_1, a\_1, r\_1, s\_2, a\_2, r\_2, ... s1​,a1​,r1​,s2​,a2​,r2​,...

Second, we compute the total reward of the trajectory R\=r1+r2+r3+...R=r\_1+r\_2+r\_3+...R\=r1​+r2​+r3​+..., and compute an estimate the gradient of the expected reward, ggg:

g\=R∑t∇θlogπθ(at∣st) g = R \\sum\_t \\nabla\_\\theta \\log\\pi\_\\theta(a\_t|s\_t) g\=Rt∑​∇θ​logπθ​(at​∣st​)

Third, we update our policy using gradient ascent with learning rate α\\alphaα:

θ←θ+αg \\theta \\leftarrow \\theta + \\alpha g θ←θ+αg

The process then repeats.

What are the main problems of REINFORCE? There are three issues:

1. The update process is very **inefficient**! We run the policy once, update once, and then throw away the trajectory.
    
2. The gradient estimate ggg is very **noisy**. By chance the collected trajectory may not be representative of the policy.
    
3. There is no clear **credit assignment**. A trajectory may contain many good/bad actions and whether these actions are reinforced depends only on the final total output.
    

In the following concepts, we will go over ways to improve the REINFORCE algorithm and resolve all 3 issues. All of the improvements will be utilized and implemented in the PPO algorithm.



# Noise Reduction

The way we optimize the policy is by maximizing the average rewards U(θ)U(\\theta)U(θ). To do that we use stochastic gradient ascent. Mathematically, the gradient is given by an average over all the possible trajectories,

∇θU(θ)\=∑τP(τ;θ)⏞average overall trajectories(Rτ∑t∇θlogπθ(at(τ)∣st(τ)))⎵only one is sampled \\nabla\_\\theta U(\\theta) = \\overbrace{\\sum\_\\tau P(\\tau; \\theta)}^{ \\begin{matrix} \\scriptsize\\textrm{average over}\\\\ \\scriptsize\\textrm{all trajectories} \\end{matrix} } \\underbrace{\\left( R\_\\tau \\sum\_t \\nabla\_\\theta \\log \\pi\_\\theta(a\_t^{(\\tau)}|s\_t^{(\\tau)}) \\right)}\_{ \\textrm{only one is sampled} } ∇θ​U(θ)\=τ∑​P(τ;θ) ​average overall trajectories​​only one is sampled (Rτ​t∑​∇θ​logπθ​(at(τ)​∣st(τ)​))​​

There could easily be well over millions of trajectories for simple problems, and infinite for continuous problems.

For practical purposes, we simply take one trajectory to compute the gradient, and update our policy. So a lot of times, the result of a sampled trajectory comes down to chance, and doesn't contain that much information about our policy. How does learning happen then? The hope is that after training for a long time, the tiny signal accumulates.

The easiest option to reduce the noise in the gradient is to simply sample more trajectories! Using distributed computing, we can collect multiple trajectories in parallel, so that it won’t take too much time. Then we can estimate the policy gradient by averaging across all the different trajectories

st(1),at(1),rt(1)st(2),at(2),rt(2)st(3),at(3),rt(3)⋮}→g\=1N∑i\=1NRi∑t∇θlogπθ(at(i)∣st(i)) \\left. \\begin{matrix} s^{(1)}\_t, a^{(1)}\_t, r^{(1)}\_t\\\\\[6pt\] s^{(2)}\_t, a^{(2)}\_t, r^{(2)}\_t\\\\\[6pt\] s^{(3)}\_t, a^{(3)}\_t, r^{(3)}\_t\\\\\[6pt\] \\vdots \\end{matrix} \\;\\; \\right\\}\\!\\!\\!\\! \\rightarrow g = \\frac{1}{N}\\sum\_{i=1}^N R\_i \\sum\_t\\nabla\_\\theta \\log \\pi\_\\theta(a^{(i)}\_t | s^{(i)}\_t) st(1)​,at(1)​,rt(1)​st(2)​,at(2)​,rt(2)​st(3)​,at(3)​,rt(3)​⋮​⎭⎪⎪⎪⎪⎪⎪⎬⎪⎪⎪⎪⎪⎪⎫​→g\=N1​i\=1∑N​Ri​t∑​∇θ​logπθ​(at(i)​∣st(i)​)

# Rewards Normalization

There is another bonus for running multiple trajectories: we can collect all the total rewards and get a sense of how they are distributed.

In many cases, the distribution of rewards shifts as learning happens. Reward = 1 might be really good in the beginning, but really bad after 1000 training episode.

Learning can be improved if we normalize the rewards, where μ\\muμ is the mean, and σ\\sigmaσ the standard deviation.

Ri←Ri−μσμ\=1N∑iNRiσ\=1N∑i(Ri−μ)2 R\_i \\leftarrow \\frac{R\_i -\\mu}{\\sigma} \\qquad \\mu = \\frac{1}{N}\\sum\_i^N R\_i \\qquad \\sigma = \\sqrt{\\frac{1}{N}\\sum\_i (R\_i - \\mu)^2} Ri​←σRi​−μ​μ\=N1​i∑N​Ri​σ\=N1​i∑​(Ri​−μ)2 ​

(when all the RiR\_iRi​ are the same, σ\=0\\sigma =0σ\=0, we can set all the normalized rewards to 0 to avoid numerical problems)

This batch-normalization technique is also used in many other problems in AI (e.g. image classification), where normalizing the input can improve learning.

Intuitively, normalizing the rewards roughly corresponds to picking half the actions to encourage/discourage, while also making sure the steps for gradient ascents are not too large/small.



# Credit Assignment

Going back to the gradient estimate, we can take a closer look at the total reward RRR, which is just a sum of reward at each step R\=r1+r2+...+rt−1+rt+...R=r\_1+r\_2+...+r\_{t-1}+r\_t+...R\=r1​+r2​+...+rt−1​+rt​+...

g\=∑t(...+rt−1+rt+...)∇θlogπθ(at∣st) g=\\sum\_t (...+r\_{t-1}+r\_{t}+...)\\nabla\_{\\theta}\\log \\pi\_\\theta(a\_t|s\_t) g\=t∑​(...+rt−1​+rt​+...)∇θ​logπθ​(at​∣st​)

Let’s think about what happens at time-step ttt. Even before an action is decided, the agent has already received all the rewards up until step t−1t-1t−1. So we can think of that part of the total reward as the reward from the past. The rest is denoted as the future reward.

(...+rt−1⏞Rtpast+rt+...⏞Rtfuture) (\\overbrace{...+r\_{t-1}}^{\\cancel{R^{\\rm past}\_t}}+ \\overbrace{r\_{t}+...}^{R^{\\rm future}\_t}) (...+rt−1​ ​Rtpast​​​+rt​+... ​Rtfuture​​)

Because we have a Markov process, the action at time-step ttt can only affect the future reward, so the past reward shouldn’t be contributing to the policy gradient. So to properly assign credit to the action ata\_tat​, we should ignore the past reward. So a better policy gradient would simply have the future reward as the coefficient .

g\=∑tRtfuture∇θlogπθ(at∣st) g=\\sum\_t R\_t^{\\rm future}\\nabla\_{\\theta}\\log \\pi\_\\theta(a\_t|s\_t) g\=t∑​Rtfuture​∇θ​logπθ​(at​∣st​)

## Notes on Gradient Modification

You might wonder, why is it okay to just change our gradient? Wouldn't that change our original goal of maximizing the expected reward?

It turns out that mathematically, ignoring past rewards might change the gradient for each specific trajectory, but it doesn't change the **averaged** gradient. So even though the gradient is different during training, on average we are still maximizing the average reward. In fact, the resultant gradient is less noisy, so training using future reward should speed things up!


# Pong with REINFORCE (code walkthrough)

## Additional Notes

- **The performance for the REINFORCE version may be poor. You can try training with a smaller tmax\=100 and more number of episodes=2000 to see concrete results.**
- Try normalizing your future rewards over all the parallel agents, it can speed up training
- Simpler networks might perform better than more complicated ones! The original input contains 80x80x2=12800 numbers, you might want to ensure that this number steadily decreases at each layer of the neural net.
- Training performance may be significantly _worse_ on local machines. I had worse performance training on my own windows desktop with a 4-core CPU and a GPU. This may be due to the slightly different ways the emulator is rendered. So please run the code on the workspace first before moving locally
- It may be beneficial to train multiple epochs, say first using a small tmax\=200 with 500 episodes, and then train again with tmax = 400 with 500 episodes, and then finally with a even larger tmax.
- Remember to save your policy after training!
- for a challenge, try the 'Pong-v4' environment, this includes random frameskips and takes longer to train.



# Importance Sampling

## 1\. Policy Update in REINFORCE

Let’s go back to the REINFORCE algorithm. We start with a policy, πθ\\pi\_\\thetaπθ​, then using that policy, we generate a trajectory (or multiple ones to reduce noise) (st,at,rt)(s\_t, a\_t, r\_t)(st​,at​,rt​). Afterward, we compute a policy gradient, ggg, and update θ′←θ+αg\\theta' \\leftarrow \\theta + \\alpha gθ′←θ+αg.

At this point, the trajectories we’ve just generated are simply thrown away. If we want to update our policy again, we would need to generate new trajectories once more, using the updated policy.

You might ask, why is all this necessary? It’s because we need to compute the gradient for the current policy, and to do that the trajectories need to be representative of the current policy.

But this sounds a little wasteful. What if we could somehow recycle the old trajectories, by modifying them so that they are representative of the new policy? So that instead of just throwing them away, we recycle them!

Then we could just reuse the recycled trajectories to compute gradients, and to update our policy, again, and again. This would make updating the policy a lot more efficient. So, how exactly would that work?

## 2\. Importance Sampling

This is where importance sampling comes in. Let’s look at the trajectories we generated using the policy πθ \\pi\_\\thetaπθ​. It had a probability P(τ;θ) P(\\tau;\\theta)P(τ;θ), to be sampled.

Now Just by chance, the same trajectory can be sampled under the new policy, with a different probability P(τ;θ′) P(\\tau;\\theta')P(τ;θ′)

Imagine we want to compute the average of some quantity, say f(τ) f(\\tau)f(τ). We could simply generate trajectories from the new policy, compute f(τ) f(\\tau)f(τ) and average them.

Mathematically, this is equivalent to adding up all the f(τ) f(\\tau)f(τ), weighted by a probability of sampling each trajectory under the new policy.

∑τP(τ;θ′)f(τ) \\sum\_\\tau P(\\tau;\\theta') f(\\tau) τ∑​P(τ;θ′)f(τ)

Now we could modify this equation, by multiplying and dividing by the same number, P(τ;θ) P(\\tau;\\theta)P(τ;θ) and rearrange the terms.

∑τP(τ;θ)⏞sampling underold policy πθP(τ;θ′)P(τ;θ)⏞re-weightingfactorf(τ) \\sum\_\\tau \\overbrace{P(\\tau;\\theta)}^{ \\begin{matrix} \\scriptsize \\textrm{sampling under}\\\\ \\scriptsize \\textrm{old policy } \\pi\_\\theta \\end{matrix} } \\overbrace{\\frac{P(\\tau;\\theta')}{P(\\tau;\\theta)}}^{ \\begin{matrix} \\scriptsize \\textrm{re-weighting}\\\\ \\scriptsize \\textrm{factor} \\end{matrix} } f(\\tau) τ∑​P(τ;θ) ​sampling underold policy πθ​​​P(τ;θ)P(τ;θ′)​ ​re-weightingfactor​​f(τ)

It doesn’t look we’ve done much. But written in this way, we can reinterpret the first part as the coefficient for sampling under the old policy, with an extra re-weighting factor, in addition to just averaging.

Intuitively, this tells us we can use old trajectories for computing averages for new policy, as long as we add this extra re-weighting factor, that takes into account how under or over–represented each trajectory is under the new policy compared to the old one.

The same tricks are used frequently across statistics, where the re-weighting factor is included to un-bias surveys and voting predictions.

## 3\. The re-weighting factor

Now Let’s a closer look at the re-weighting factor.

P(τ;θ′)P(τ;θ)\=πθ′(a1∣s1)πθ′(a2∣s2)πθ′(a3∣s3)...πθ(a1∣s1)πθ(a2∣s2)πθ(a2∣s2)... \\frac{P(\\tau;\\theta')}{P(\\tau;\\theta)} =\\frac {\\pi\_{\\theta'}(a\_1|s\_1)\\, \\pi\_{\\theta'}(a\_2|s\_2)\\, \\pi\_{\\theta'}(a\_3|s\_3)\\,...} {\\pi\_\\theta(a\_1|s\_1) \\, \\pi\_\\theta(a\_2|s\_2)\\, \\pi\_\\theta(a\_2|s\_2)\\, ...} P(τ;θ)P(τ;θ′)​\=πθ​(a1​∣s1​)πθ​(a2​∣s2​)πθ​(a2​∣s2​)...πθ′​(a1​∣s1​)πθ′​(a2​∣s2​)πθ′​(a3​∣s3​)...​

Because each trajectory contains many steps, the probability contains a chain of products of each policy at different time-step.

This formula is a bit complicated. But there is a bigger problem. When some of policy gets close to zero, the re-weighting factor can become close to zero, or worse, close to 1 over 0 which diverges to infinity.

When this happens, the re-weighting trick becomes unreliable. So, In practice, we want to make sure the re-weighting factor is not too far from 1 when we utilize importance sampling



# PPO Part 1: The Surrogate Function

## Re-weighting the Policy Gradient

Suppose we are trying to update our current policy, πθ′\\pi\_{\\theta'}πθ′​. To do that, we need to estimate a gradient, ggg. But we only have trajectories generated by an older policy πθ\\pi\_{\\theta}πθ​. How do we compute the gradient then?

Mathematically, we could utilize importance sampling. The answer just what a normal policy gradient would be, times a re-weighting factor P(τ;θ′)/P(τ;θ)P(\\tau;\\theta')/P(\\tau;\\theta)P(τ;θ′)/P(τ;θ):

g\=P(τ;θ′)P(τ;θ)∑t∇θ′πθ′(at∣st)πθ′(at∣st)Rtfuture g=\\frac{P(\\tau; \\theta')}{P(\\tau; \\theta)}\\sum\_t \\frac{\\nabla\_{\\theta'} \\pi\_{\\theta'}(a\_t|s\_t)}{\\pi\_{\\theta'}(a\_t|s\_t)}R\_t^{\\rm future} g\=P(τ;θ)P(τ;θ′)​t∑​πθ′​(at​∣st​)∇θ′​πθ′​(at​∣st​)​Rtfuture​

We can rearrange these equations, and the re-weighting factor is just the product of all the policy across each step -- I’ve picked out the terms at time-step ttt here. We can cancel some terms, but we're still left with a product of the policies at different times, denoted by ".........".

g\=∑t...πθ′(at∣st)......πθ(at∣st)...∇θ′πθ′(at∣st)πθ′(at∣st)Rtfuture g=\\sum\_t \\frac{...\\, \\cancel{\\pi\_{\\theta'}(a\_t|s\_t)} \\,...} {...\\,\\pi\_{\\theta}(a\_t|s\_t)\\,...} \\, \\frac{\\nabla\_{\\theta'} \\pi\_{\\theta'}(a\_t|s\_t)}{\\cancel{\\pi\_{\\theta'}(a\_t|s\_t)}}R\_t^{\\rm future} g\=t∑​...πθ​(at​∣st​)......πθ′​(at​∣st​)​...​πθ′​(at​∣st​)​∇θ′​πθ′​(at​∣st​)​Rtfuture​

Can we simplify this expression further? This is where proximal policy comes in. If the old and current policy is close enough to each other, all the factors inside the "........." would be pretty close to 1, and then we can ignore them.

Then the equation simplifies

g\=∑t∇θ′πθ′(at∣st)πθ(at∣st)Rtfuture g=\\sum\_t \\frac{\\nabla\_{\\theta'} \\pi\_{\\theta'}(a\_t|s\_t)}{\\pi\_{\\theta}(a\_t|s\_t)}R\_t^{\\rm future} g\=t∑​πθ​(at​∣st​)∇θ′​πθ′​(at​∣st​)​Rtfuture​

It looks very similar to the old policy gradient. In fact, if the current policy and the old policy is the same, we would have exactly the vanilla policy gradient. But remember, this expression is different because we are comparing two _different_ policies

## The Surrogate Function

Now that we have the approximate form of the gradient, we can think of it as the gradient of a new object, called the surrogate function

g\=∇θ′Lsur(θ′,θ) g=\\nabla\_{\\theta'} L\_{\\rm sur}(\\theta', \\theta) g\=∇θ′​Lsur​(θ′,θ)

Lsur(θ′,θ)\=∑tπθ′(at∣st)πθ(at∣st)Rtfuture L\_{\\rm sur}(\\theta', \\theta)= \\sum\_t \\frac{\\pi\_{\\theta'}(a\_t|s\_t)}{\\pi\_{\\theta}(a\_t|s\_t)}R\_t^{\\rm future} Lsur​(θ′,θ)\=t∑​πθ​(at​∣st​)πθ′​(at​∣st​)​Rtfuture​

So using this new gradient, we can perform gradient ascent to update our policy -- which can be thought as directly maximize the surrogate function.

But there is still one important issue we haven’t addressed yet. If we keep reusing old trajectories and updating our policy, at some point the new policy might become different enough from the old one, so that all the approximations we made could become invalid.

We need to find a way make sure this doesn’t happen. Let’s see how in part 2.


# PPO Part 2: Clipping Policy Updates

## The Policy/Reward Cliff

What is the problem with updating our policy and ignoring the fact that the approximations are not valid anymore? One problem is it could lead to a really bad policy that is very hard to recover from. Let's see how:

![](https://video.udacity-data.com/topher/2018/September/5b9a9625_policy-reward-cliff/policy-reward-cliff.png)

Say we have some policy parameterized by πθ′\\pi\_{\\theta'}πθ′​ (shown on the left plot in black), and with an average reward function (shown on the right plot in black).

The current policy is labelled by the red text, and the goal is to update the current policy to the optimal one (green star). To update the policy we can compute a surrogate function LsurL\_{\\rm sur}Lsur​ (dotted-red curve on right plot). So LsurL\_{\\rm sur}Lsur​ approximates the reward pretty well around the current policy. But far away from the current policy, it diverges from the actual reward.

If we continually update the policy by performing gradient ascent, we might get something like the red-dots. The big problem is that at some point we hit a cliff, where the policy changes by a large amount. From the perspective of the surrogate function, the average reward is really great. But the actually average reward is really bad!

What’s worse, the policy is now stuck in a deep and flat bottom, so that future updates won’t be able to bring the policy back up! we are now stuck with a really bad policy.

How do we fix this? Wouldn’t it be great if we can somehow stop the gradient ascent so that our policy doesn’t fall off the cliff?

## Clipped Surrogate Function

![](https://video.udacity-data.com/topher/2018/September/5b9a99cd_clipped-surrogate/clipped-surrogate.png)

Here’s an idea: what if we just flatten the surrogate function (blue curve)? What would policy update look like then?

So starting with the current policy (blue dot), we apply gradient ascent. The updates remain the same, until we hit the flat plateau. Now because the reward function is flat, the gradient is zero, and the policy update will stop!

Now, keep in mind that we are only showing a 2D figure with one θ′\\theta'θ′ direction. In most cases, there are thousands of parameters in a policy, and there may be hundreds/thousands of high-dimensional cliffs in many different directions. We need to apply this clipping mathematically so that it will automatically take care of all the cliffs.

## Clipped Surrogate Function

Here's the formula that will automatically flatten our surrogate function to avoid all the cliffs:

Lsurclip(θ′,θ)\=∑tmin{πθ′(at∣st)πθ(at∣st)Rtfuture,clipϵ(πθ′(at∣st)πθ(at∣st))Rtfuture} L^{\\rm clip}\_{\\rm sur} (\\theta',\\theta)= \\sum\_t \\min\\left\\{ \\frac{\\pi\_{\\theta'}(a\_t|s\_t)}{\\pi\_{\\theta}(a\_t|s\_t)}R\_t^{\\rm future} , {\\rm clip}\_\\epsilon\\!\\! \\left( \\frac{\\pi\_{\\theta'}(a\_t|s\_t)} {\\pi\_{\\theta}(a\_t|s\_t)} \\right) R\_t^{\\rm future} \\right\\} Lsurclip​(θ′,θ)\=t∑​min{πθ​(at​∣st​)πθ′​(at​∣st​)​Rtfuture​,clipϵ​(πθ​(at​∣st​)πθ′​(at​∣st​)​)Rtfuture​}

Now let’s dissect the formula by looking at one specific term in the sum, and set the future reward to 1 to make things easier.

![](https://video.udacity-data.com/topher/2018/September/5b9a9d58_clipped-surrogate-explained/clipped-surrogate-explained.png)

We start with the original surrogate function (red), which involves the ratio πθ′(at∣st)/πθ(at∣st)\\pi\_{\\theta'}(a\_t|s\_t)/\\pi\_\\theta(a\_t|s\_t)πθ′​(at​∣st​)/πθ​(at​∣st​). The black dot shows the location where the current policy is the same as the old policy (θ′\=θ\\theta'=\\thetaθ′\=θ)

We want to make sure the two policy is similar, or that the ratio is close to 1. So we choose a small ϵ\\epsilonϵ (typically 0.1 or 0.2), and apply the clip{\\rm clip}clip function to force the ratio to be within the interval \[1−ϵ,1+ϵ\]\[1-\\epsilon,1+\\epsilon\]\[1−ϵ,1+ϵ\] (shown in purple).

Now the ratio is clipped in two places. But we only want to clip the top part and not the bottom part. To do that, we compare this clipped ratio to the original one and take the minimum (show in blue). This then ensures the clipped surrogate function is always less than the original surrogate function Lsurclip≤LsurL\_{\\rm sur}^{\\rm clip}\\le L\_{\\rm sur}Lsurclip​≤Lsur​, so the clipped surrogate function gives a more conservative "reward".

(_the blue and purple lines are shifted slightly for easier viewing_)



# PPO Summary

So that’s it! We can finally summarize the PPO algorithm

1. First, collect some trajectories based on some policy πθ\\pi\_\\thetaπθ​, and initialize theta prime θ′\=θ\\theta'=\\thetaθ′\=θ
2. Next, compute the gradient of the clipped surrogate function using the trajectories
3. Update θ′\\theta'θ′ using gradient ascent θ′←θ′+α∇θ′Lsurclip(θ′,θ)\\theta'\\leftarrow\\theta' +\\alpha \\nabla\_{\\theta'}L\_{\\rm sur}^{\\rm clip}(\\theta', \\theta)θ′←θ′+α∇θ′​Lsurclip​(θ′,θ)
4. Then we repeat step 2-3 without generating new trajectories. Typically, step 2-3 are only repeated a few times
5. Set θ\=θ′\\theta=\\theta'θ\=θ′, go back to step 1, repeat.

_The details of PPO was originally published by the team at OpenAI, and you can read their paper through this [link](https://arxiv.org/abs/1707.06347)_



# Pong with PPO (code walkthrough)

## Additional Notes

- Try normalizing your future rewards over all the parallel agents, it can speed up training
- Simpler networks might perform better than more complicated ones! The original input contains 80x80x2=12800 numbers, you might want to ensure that this number steadily decreases at each layer of the neural net.
- Training performance may be significantly _worse_ on local machines. I had worse performance training on my own windows desktop with a 4-core CPU and a GPU. This may be due to the slightly different ways the emulator is rendered. So please run the code on the workspace first before moving locally
- It may be beneficial to train multiple epochs, say first using a small tmax\=200 with 500 episodes, and then train again with tmax = 400 with 500 episodes, and then finally with a even larger tmax.
- Remember to save your policy after training!
- for a challenge, try the 'Pong-v4' environment, this includes random frameskips and takes longer to train.


