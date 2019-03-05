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

