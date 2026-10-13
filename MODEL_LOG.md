# MODEL_LOG

One block per model family (assignment Section 9.1). Copy the block below.

## <Model name>, Chapter <n>, depth <A/B/C>

- Objective / factorization / decision rule:
- Assumptions, inductive bias, expected failure modes:
- Representation and preprocessing:
- Own pre-reference commit; reference code consulted (repository, file, function, commit or version):
- Hyperparameters and selection procedure:
- Primary / secondary metrics, runtime, model size:
- Diagnostic plot or table; focused experiment:
- Error and limitation analysis; use-case fit:
- Exam-ready paragraph (no code):

## Decision tree, Chapter 2, depth B

- Objective / factorization / decision rule: Maximise Information Gain at each branching point; the decision rule is to follow the conditional path to a leaf node and return the majority class label.
- Assumptions, inductive bias, expected failure modes: Assumes that decision boundaries are orthogonal to the coordinate axes; the inductive bias favours short trees with high impurity reduction near the root; the model is prone to overfitting if depth is not controlled or pruning is not applied when the data contains substantial noise.
- Representation and preprocessing: 561 time-domain and frequency-domain features from accelerometer and gyroscope sensors; feature scaling is not required because trees are invariant to monotonic transformations; the data is split by subject.
- Own pre-reference commit; reference code consulted (repository, file, function, commit or version): Initial commit `3d79af3`; consulted `scikit-learn DecisionTreeClassifier`.
- Hyperparameters and selection procedure: `max_depth=None`, `random_state=31`, and the Gini split criterion; checked the actual depth and number of leaf nodes on the training set.
- Primary / secondary metrics, runtime, model size: Train Accuracy ~ 1.00, Val Accuracy ~ 0.81; maximum depth 20, 139 leaf nodes.
- Diagnostic plot or table; focused experiment: Feature importance analysis: the three most important features are (Columns 558, 86, and 9).
- Error and limitation analysis; use-case fit: A single decision tree is unstable and tends to confuse static activities (Sitting vs Standing). Although inference is very fast and lightweight for a phone, its overall accuracy is lower than that of a deep nonlinear model.
- Exam-ready paragraph (no code): A CART decision tree partitions the feature space into axis-aligned rectangular regions by selecting features and split thresholds that optimise the reduction in Gini impurity. The algorithm uses a greedy top-down strategy and produces a highly interpretable model. However, its axis-parallel split bias makes it difficult to represent complex cross-feature correlations without feature combinations or an ensemble method.
## Multilayer Perceptron (MLP) & Perceptron, Chapter 3, depth A
- Objective / factorization / decision rule: Minimise the mean cross-entropy loss: $$L = -(1/N) \sum_i \sum_k Y_{ik} \log P_{ik}$$ with a Softmax output layer; the classification rule is $$\hat{y} = \arg\max_k P_k$$ use backpropagation and the chain rule to update weights W and biases b.
- Assumptions, inductive bias, expected failure modes: Assumes that nonlinear relationships among sensor signals can be approximated by compositions of ReLU activation functions; the inductive bias favours smooth functions learned through Stochastic Gradient Descent; the model may fail when unnormalised data causes vanishing or exploding gradients, or when a learning rate that is too large overshoots optima.
- Representation and preprocessing: 561 real-valued features; mandatory standardisation using `StandardScaler` (mean=0, std=1); subject-aware splitting prevents data leakage between the training and validation sets.
- Own pre-reference commit; reference code consulted (repository, file, function, commit or version): First self-coded commit `87c5fbe`.
- Hyperparameters and selection procedure: `random_state=31`, and `max_iter=561`. Compared a small hidden layer model (10 neurons) with a large one (200 neurons) on the validation set.
- Primary / secondary metrics, runtime, model size:
  - Model A (10 hidden neurons): Training Score: 1.0000, Validation Score: 0.9355, runtime ~ 5.26 seconds.
  - Model B (200 hidden neurons): Training Score: 1.0000, Validation Score: 0.9326, runtime ~ 4.7916 seconds.
- Diagnostic plot or table; focused experiment: Model capacity comparison: Model A improves validation accuracy by ~0.31% compared with Model B.
- Error and limitation analysis; use-case fit: The remaining errors are mainly confusion between Walking Upstairs and Walking Downstairs when the sensor-motion amplitudes are similar. Model A (10 units) is compact and lightweight, making it well suited for direct real-time deployment on a smartphone processor.
- Exam-ready paragraph (no code): A multilayer perceptron (MLP) overcomes the linear-separation limitation of a single-layer Perceptron by stacking affine transformations with nonlinear activation functions such as ReLU. Backpropagation applies the chain rule to compute the partial derivative of the cross-entropy loss with respect to each weight, from the output layer back to the input layer. Gradient-based optimisation adjusts the decision hyperplanes to separate complex data classes in a high-dimensional space.

## Naive Bayes, Chapter 4, depth A
Gaussian Naive Bayes was applied to dataset as a probabilistic baseline, utilizing the 561 time-domain and frequency-domain features from the accelerometer and gyroscope sensors. While it achieved a modest validation accuracy of 64.00%, empirical evidence shows the model is severely miscalibrated, highlighted by an exceptionally high Log Loss of 10.6256. By diagnosing the worst-case failure sample (index 13), I observed that the model predicted class 4 instead of the true class 5 with an absolute predicted probability of 1.0000. This extreme overconfidence in an incorrect prediction demonstrates a severe violation of the algorithm's core conditional independence assumption. Because multi-axis smartphone sensor readings are naturally highly correlated, the model artificially compounds the evidence from these dependent features.