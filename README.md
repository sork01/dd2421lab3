# LAB.3 - Bayesian Learning & Boosting

## Assignment 1

~~~~
def mlParams(X, labels, W):
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses, Ndims))
    sigma = np.zeros((Nclasses, Ndims, Ndims))

    for jdx, c in enumerate(classes):
        idx = np.where(labels == c)[0]  
        # get the x for the class labels. vectors are rows                            
        xlc = X[idx]                                                         
        xlw = W[idx]
        # compute mean
        mu[jdx] = sum(xlw * xlc) / sum(xlw)        
    for jdx, c in enumerate(classes):
        idx = np.where(labels == c)[0]                     
        xlc = np.array(X[idx])
        xlw = W[idx]                                                  
        u = np.array(mu[jdx])
        # diff between x and mu
        x = xlc - u
        # compute mean
        mean = sum(xlw * np.square(x)) / sum(xlw)
        # diagonal matrix since "0 for m != n"
        sigma[jdx] = np.diag(mean)                         

    return mu, sigma
~~~~

<p align="center"><img src="https://github.com/sork01/dd2421lab3/blob/master/ass1.png"></p>


## Assignment 2

computePrior
~~~~
# estimates and returns the class prior in X
def computePrior(labels, W):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))

    for jdx, c in enumerate(classes):
        idx = np.where(labels == c)[0]
        xlw = W[idx]
        prior[jdx] = np.sum(xlw) / np.sum(W)

    return prior
~~~~

classifyBayes
~~~~
def classifyBayes(X, prior, mu, sigma):

    Npts = X.shape[0]
    Nclasses,Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))
    
    #-0.5(ln(determinant(sigma)) - 0.5(x* - mu) * inverse (x* - mu ) transpose + ln(class prior)
    for jdx in range(Nclasses):
        halfdet = (-1/2) * np.log(np.linalg.det(sigma[jdx]))
        inverse = np.linalg.inv(sigma[jdx])
        logpost = np.log(prior[jdx])

        for i, x in enumerate(X):
            xminusmu = x - mu[jdx]
            logProb[jdx][i] = halfdet - (1/2) * np.dot(np.dot(xminusmu, inverse), np.transpose(xminusmu)) + logpost

    # one possible way of finding max a-posteriori once
    # you have computed the log posterior
    h = np.argmax(logProb,axis=0)
    return h
~~~~

## Assignment 3

### Iris

Trial | Accuracy
--- | ---
0 | 84.4
10 | 95.6
20 | 93.3
30 | 86.7
40 | 88.9
50 | 91.1
60 | 86.7
70 | 91.1
80 | 86.7
90 | 91.1

*Final mean classification accuracy  89 with standard deviation 4.16

Plot boundary:

<p align="center"><img src="https://github.com/sork01/dd2421lab3/blob/master/ass3.png"></p>

### Vowel

Trial | Accuracy
--- | ---
0 | 61
10 | 66.2
20 | 74
30 | 66.9
40 | 59.7
50 | 64.3
60 | 66.9
70 | 63.6
80 | 62.3
90 | 70.8

*Final mean classification accuracy  64.7 with standard deviation 4.03


1) **When can a feature independence assumption be reasonable and when not?**
In real life features are more often not independent, most of the time the opposite is true. For example when predicting weather, NB would assume that the humidity is not dependent on the cloud cover. This is why it's called Naive Bayes, since independent features is a pretty big and often false assumption. 
However even with dependent features NB is a fairly good classifyer but a poor probability estimator.

2) **How does the decision boundary look for the Iris dataset? How could one improve the classiﬁcation results for this scenario by changing classifier or, alternatively, manipulating the data?**
The classes 0 and 1 are beautifully divided since they're easily separable classes, classes 1 and 2 are somewhat noisy and overlapping which results in a strange boudary leaning the wrong way. A more intuitive boundary would be leaning towards the right instead. 
To improve the classification one could instead maybe use SVM with some slack or adding weights.


## Assignment 5

trainBoost
~~~~
def trainBoost(base_classifier, X, labels, T):
    Npts,Ndims = np.shape(X)

    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # do classification for each point
        vote = classifiers[-1].classify(X)

        ht = np.zeros((Npts,1))
        # initialize all weights of your samples to 1 divided by number of training sample
        weight = np.sum(wCur)
        for i in range(Npts):
            if (vote[i] == labels[i]):
                ht[i] = 1
            else:
                ht[i] = 0
        
        error = np.sum(wCur * (1 - ht))
        alphat = 0.5 * np.log((1-error) - np.log(error)) 
        alphas.append(alphat)
        oldw = wCur
        for i in range(Npts):
            if (ht[i] == 1):
                wCur[i] = oldw[i] * np.exp(-alphat)
            else:
                wCur[i] = oldw[i] * np.exp(alphat)            
        
    return classifiers, alphas

~~~~

classifyBoost

~~~~
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))
        
        for i in range(Ncomps):
            test = classifiers[i].classify(X)
            for j in range(Npts):
                # alphas is weighted votes
                votes[j][test[j]] += alphas[i]

        # one way to compute yPred after accumulating the votes
        return np.argmax(votes,axis=1)
~~~~

### Iris

Trial | Accuracy
--- | ---
0 | 97.8
10 | 97.8
20 | 91.1
30 | 91.1
40 | 97.8
50 | 93.3
60 | 88.9
70 | 93.3
80 | 01.1
90 | 91.1

*Final mean classification accuracy  93.4 with standard deviation 3.39

<p align="center"><img src="https://github.com/sork01/dd2421lab3/blob/master/ass5.png"></p>

### Vowel

Trial | Accuracy
--- | ---
0 | 76
10 | 83.1
20 | 81.2
30 | 71.4
40 | 69.5
50 | 72.7
60 | 81.8
70 | 82.5
80 | 77.9
90 | 78.6

*Final mean classification accuracy  77.9 with standard deviation 3.44


1) Is there any improvement in classification accuracy? Why/why not?

 &nbsp;| Iris | Vowel
--- | --- | ---
**Before boost** | 89 |  64,7
**After boost** | 93,4 | 77,9
**Difference** | 4,4 | 13,2

Both sets had an improvement in classification accuracy, this i because the boosting concentrate on the missclassified datapoints and adding weights to them thus increasing the variance of the model


2) Plot the decision boundary of the boosted classiﬁer on iris and compare it with that of the basic. What diﬀerences do you notice? Is the boundary of the boosted version more complex

The boundary between class 0 and 1 in the boosted version is a little less complex, however the boundary between class 1 and 2 is a lot more complex. It fits the data more accurately

3) Can we make up for not using a more advanced model in the basic classiﬁer (e.g. independent features) by using boosting?

Yes, boosting can increase accuracy of the basic classifier



## Assignment 6

### Decision Tree Classifier Iris

Trial | Accuracy
--- | ---
0 | 95.6
10 | 100
20 | 91.1
30 | 91.1
40 | 93.3
50 | 91.1
60 | 88.9
70 | 88.9
80 | 93.3
90 | 88.9

*Final mean classification accuracy  92.4 with standard deviation 3.71

<p align="center"><img src="https://github.com/sork01/dd2421lab3/blob/master/ass7_1.png"></p>

### Decision Tree Boosted Iris

Trial | Accuracy
--- | ---
0 | 95.6
10 | 95.6
20 | 97.8
30 | 93.3
40 | 88.9
50 | 93.3
60 | 93.3
70 | 91.1
80 | 97.8
90 | 97.8

*Final mean classification accuracy  94.2 with standard deviation 3.22

<p align="center"><img src="https://github.com/sork01/dd2421lab3/blob/master/ass7_2.png"></p>

### Decision Tree Classifier Vowel

Trial | Accuracy
--- | ---
0 | 63.6
10 | 68.8
20 | 63.6
30 | 66.9
40 | 59.7
50 | 63
60 | 59.7
70 | 68.8
80 | 59.7
90 | 68.2

*Final mean classification accuracy  64.1 with standard deviation 4


### Decision Tree Boosted Vowel

Trial | Accuracy
--- | ---
0  |82.5
10 | 86.4
20 | 86.4
30 | 92.2
40 | 88.3
50 | 84.4
60 | 85.1
70 | 89
80 | 83.8
90 | 88.3

*Final mean classification accuracy  86.4 with standard deviation 2.71

1) Is there any improvement in classification accuracy? Why/why not?

 &nbsp;| Iris | Vowel
--- | --- | ---
**Before boost** | 92,4 |  64,1
**After boost** | 94,2 | 86,4
**Difference** | 1,8 | 22,3

Both sets had an improvement in classification accuracy, however Vowel had a huge improvement, i'm guessing it's because vowel has more datapoints.


2) Plot the decision boundary of the boosted classiﬁer on iris and compare it with that of the basic. What diﬀerences do you notice? Is the boundary of the boosted version more complex

The boundary in the boosted version is more complex.

3) Can we make up for not using a more advanced model in the basic classiﬁer (e.g. independent features) by using boosting?

Yes, boosting can increase accuracy of some models, in this case the vowels dataset gained a lot of accuracy. The iris set didn't gain that much.


## Assignment 7

** If you had to pick a classiﬁer, naive Bayes or a decision tree or the boosted versions of these, which one would you pick? Motivate from the following criteria: ** 



###Outliers
    
Naive bayes with no boosting since boosting will give larger weights to outliers and decision tree usually overfit the data

###Irrelevant input

Decision tree, pruning will remove most of the irrelevant input. However if the features are independent naive bayes would work aswell

###Predictive power

I'm guessing that it depends on if the features are independent or not. If the features are independent, Naive bayes would be best, if not Decision tree. Both boosted to get better accuracy

###Mixed types of data

Decision tree is more flexible when it comes to mixed type of data, using boosting would improve accuracy I think

###Scalability 

When the datasets gets bigger, there will probably be more and more non independent features so Decision tree would be better. No boosting since it would be too expensive.
