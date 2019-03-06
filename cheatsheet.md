# Cheat-sheet Machine learning

## Decision Trees

Tree models where the target variable can take a discrete set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees.

Information gain is used to decide which feature to split on at each step in building the tree. Simplicity is best, so we want to keep our tree small. 

### Pros

**Simple to understand and interpret** 

**Able to handle both numerical and categorical data**

**Requires little data preparation**

**Performs well with large datasets**

**Mirrors human decision making more closely than other approaches**

### Cons 

**Trees can be very non-robust. A small change in the training data can result in a large change in the tree and consequently the final predictions**

**Overfitting (could be countered with pruning)**

## Probabilistic Learning

A latent variable is a hidden variable that are not directly observed, one advantage of using latent variables is that they can serve to reduce the dimensionality of data.

## Naive Bayes Classifier

All naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable. For example, a fruit may be considered to be an apple if it is red, round, and about 10 cm in diameter. A naive Bayes classifier considers each of these features to contribute independently to the probability that this fruit is an apple, regardless of any possible correlations between the color, roundness, and diameter features.

### Posterior probability

A posterior probability, in Bayesian statistics, is the revised or updated probability of an event occurring after taking into consideration new information. Conditional probability taking into account the evidence



## Neural Networks

### Backpropagation

Backpropagation is an advanced algorithm, driven by sophisticated mathematics, which allows us to adjust all the weights in our Neural Network.
The key underlying principle of Backpropagation is that the structure of the algorithm allows for large numbers of weights to be adjusted simultaneously.

### Dropout

Dropout is when you're dropping out units in a neural network, ignoring units/neurons during the training phase at random to prevent overfitting.

## Support Vector Machines

Basic idea of support vector machines: just like 1-layer or multi-layer neural nets.
Support vectors are the data points that lie closest to the decision surface (or hyperplane)which are the datapoints that are the most difficult to classify.
SVMs maximize the margin around the separating hyperplane.
The decision function is fully specified by a (usually very small) subset of training  samples, the support vectors.
If a training sample has a high alpha value it means that that sample has a large influence on the resulting classifier.

### Non-Linear SVM

The idea is to gain linearly separation by mapping the data to a higher dimensional space

### Kernel 

writing here soon

## Ensemble Learning

Ensemble methods use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone.
Ensemble is combining diverse set of learners (individual models) together to improvise on the stability and predictive power of the model.

### Bagging 

Bagging tries to implement similar learners on small sample populations and then takes a mean of all the predictions. In generalized bagging, you can use different learners on different population. This helps to reduce the variance error.

### Boosting

Boosting is an iterative technique which adjust the weight of an observation based on the last classification. If an observation was classified incorrectly, it tries to increase the weight of this observation and vice versa. Boosting in general decreases the bias error and builds strong predictive models. However, they may sometimes overfit on the training data.

### Stacking

Use a learner to combine output from different learners. This can lead to decrease in either bias or variance error depending on the combining learner used.

## Bias

Bias error is useful to quantify how much on an average are the predicted values different from the actual value. A high bias error means we have a under-performing model which keeps on missing important trends.

## Variance 

Variance quantifies how the prediction made on same observation differs from each other. A high variance model will over-fit on your training population and perform badly on any observation beyond training.

## The Subspace Method

Principal Component Analysis (PCA) is by far one of the most popular algorithms for dimensionality reduction. Given a set of observations x , with dimension M, PCA is the standard technique for finding the single best subspace of a given dimension, m


### K-fold cross validation

Cross-validation is a statistical method used to estimate the skill of machine learning models. A technique for assessing a model while exploiting available data for training and testing

### Curse of Dimensionality

The problem is that when the dimensionality increases, the volume of the space increases so fast that the available data become sparse.

### The Lasso

Least Absolute Shrinkage and Selection Operator.
Lasso was introduced in order to improve the prediction accuracy and interpretability of regression models by altering the model fitting process to select only a subset of the provided covariates for use in the final model rather than using all of them. An approach to regression that results in feature seletion

## Perceptron Learning

Method to find separating hyperplanes

## RANSAC
Random Sample Consensus is an iterative method to estimate parameters of a mathematical model from a set of observed data that contains outliers, when outliers are to be accorded no influence on the values of the estimates. Therefore, it also can be interpreted as an outlier detection method.

## Projection Length

A similarity measure in the subspace method
