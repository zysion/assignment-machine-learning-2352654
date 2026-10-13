# W06 Bayesian Learning / Naive Bayes

## A. Concept capsule
Bayes's theroem:
$$P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}$$

$\theta$: theory, parameter of model
$D$: collected data
$P(D|\theta): likelihood
$P(\theta)$: the proportion of prior theory, the inital theory that we believe before obser dataset
$P(\theta|D)$: the posterior theory after obsere data.
$P(D)$: evidence, the set of data that obsered.

Likelihood mean that with theory $\theta$, how much percentage that we can get the result D.
The goal of bayes learning is maximize value of likelihood.

There are to way in bayes learning.
1. MLE (Maximum likelihood extimation)
Choose $\theta$ that maximize likelihood of data. It mean that choose the most sufficient $\theta$ value that can fit with the observed data.

- This method do not be affected by prior theory.
- Work well in the large train set because in large data. Because the probability curve becomes smoother in accordance with the law of large numbers and is less affected by outliers.

2. MAP (Maximum a posterior estimation)
In this method, we seek a value for $\theta$ such that, given the observed dataset D, the resulting conclusion has the highest probability. This implies that the conclusion we reach depends entirely on our observations; in other words, the method refines the theory based on the experience gained from observing the data.

- Prior $P(\theta)$ represents background knowledge or assumption.
- Useful when limit data. This method seeks the value of theta that yields the most plausible conclusion based on the observed dataset D. Consequently, when the dataset contains limited information, the model's ability to learn from past experience offers a distinct advantage.
## B. One derivation or worked example
The prevalence of a disease ($\theta = disease$) in the community is 1% ($P(\theta) = 0.01$), and a testing device yields a positive result for 90% of those who have the disease ($P(+|\theta) = 0.9$). If a patient receives a positive result, what is the probability that they actually have the disease?

We have $P(disease | +) ∝ P(+|disease) * P(disease) = 0.9 * 0.01 = 0.009$
Meanwhile, $P(no disease | +) ∝ P(+|no disease) * P(no disease) = 0.1 * 0.99 = 0.099$

=> $P(disease | +) = \frac{0.009}{0.009 + 0.099} = 0.083$
## C. Code-to-theory trace

## D. One controlled experiment

## E. Failure / misconception
I used to get trouble about the definition of likelihood, because it look like the probability. But now I know that likehood is the value that with this assumption, how trustable to the given sample.
## F. Written-exam capsule

## G. Reflection
I have gained a fresh perspective on machine learning, viewing it through the lens of probability rather than the weight-based approach of the perceptron. This lesson highlighted the vast array of machine learning algorithms available and demonstrated that selecting the right one to suit specific requirements and the problem at hand is by no means an easy task.
## H. Inquiry trail
When I study this chapter, I get trouble about the meaning of likelihood and the parameter "theory", so I ask AI to explain the meaning of likelihood and $\theta$. Then I watch the video [
Probability is not Likelihood](https://www.youtube.com/watch?v=zgir07qRdGw) to confirm and get more information to understand more likelihood's definition.
<!-- Mandatory whenever AI was used: the question you asked, the link to your pre-AI attempt commit, a summary of the
hint (not a pasted solution), the textbook/MOOC/repository source you used to verify it, and what you can now
reproduce without AI. For NPTEL videos use the 3-2-1 routine (Section 12.4) and link one exam-style question. -->
