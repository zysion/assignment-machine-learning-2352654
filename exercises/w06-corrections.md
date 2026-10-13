I do not fully understand generative and discriminative model.
# (D)
## in generative model.
The model learn $P(X,Y)$ by using the probability of prior hypothesis $P(Y)$. $$P(X,Y) = P(X|Y) \cdot P(Y)$$
$$\rightarrow P(X|Y) = \frac{P(X,Y)}{P(Y)}$$
Then use Bayes theorem to calculate posterior
$$P(Y|X) = \frac{P(X|Y) \cdot P(Y)}{P(X)}$$

## in discriminative model.
The model learn from posteriori to draw a function f that map the input to the result directly.

## compare table
| Metrics | generative | Discriminative |
|:---|:---|:---|
|with small dataset| less be overfiting and train faster | easy to be overfitting if the number of samples quite less compare to features. |
|with big dataset| high asymptotic error if The initial assumption deviated from reality. | have more accuracy because do not be stuck with any assumption. |
| Outlier | vulnerable with outlier because it affect the expected and variance | It just only focus on the vicinity of the decision boundary |