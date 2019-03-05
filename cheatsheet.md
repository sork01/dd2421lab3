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

## Neural Networks

### Backpropagation

Backpropagation is an advanced algorithm, driven by sophisticated mathematics, which allows us to adjust all the weights in our Neural Network.
The key underlying principle of Backpropagation is that the structure of the algorithm allows for large numbers of weights to be adjusted simultaneously.

## Support Vector Machines

Basic idea of support vector machines: just like 1-layer or multi-layer neural nets
