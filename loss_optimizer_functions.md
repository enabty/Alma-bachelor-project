# Introduction

## Loss function
- The loss function is a measure of how well the model is able to minimize the difference between the predicted output and the actual output on the training data. 
* The goal is to minimize the loss function during training.

## Optimizer function
- The optimizer function is responsible for updating the weights of the neural network based on the gradients of the loss function.
- The goal of the optimizer function is to find the optimal set of weights that minimizes the loss function.
---
## Model architecture
In summary, the loss function measures the error of the model's predictions, while the optimizer updates the weights of the model to minimize the loss function. Hence these two functions go hand in hand when building the architecture of a neural network. 

# Choice of functions
In order to find the optimal loss/optimizer functions for our dataset I've tried multiple combinations that are said to be suited for binary classification from various sources. Below is a summary/description of the functions tested as well as the results from each iteration further below.
## Loss functions used
Summary of the function as well as pros/cons and use case of each.
### Binary cross-entropy
- Used for binary classification problems where the output of the model is a single probability value between 0 and 1.
- Pros: Easy to implement, computationally efficient, works well for binary classification problems.
- Cons: May not work well for multi-class classification problems.
### Hinge
- Used for maximum margin binary classification problems where the goal is to maximize the margin between the decision boundary and the training data.
- Pros: Works well for maximum margin binary classification problems, can handle noisy data.
- Cons: May not work well for other types of classification problems, can be sensitive to outliers.
### Squared Hinge
- Similar to hinge loss, but with a smoother surface that can help prevent over-fitting.
- Pros: Can help prevent over-fitting, works well for maximum margin binary classification problems.
- Cons: May not work well for other types of classification problems, can be sensitive to outliers.
## Optimizer used
Summary of the function as well as pros/cons and use case of each.
### Adadelta
- Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate.
- Pros: Adapts the learning rate for each weight in the network, requires little configuration, works well for sparse data.
- Cons: Can be computationally expensive, may not work well for non-stationary problems.
### Adam
- Adam is an adaptive learning rate optimization algorithm that combines the advantages of both AdaGrad and RMSProp.
- Pros: Adapts the learning rate for each weight in the network, works well for large datasets, computationally efficient.
- Cons: Can be sensitive to hyperparameter choices, may not work well for non-stationary problems.
### SGD
- SGD is a simple optimization algorithm that updates the weights of the network based on the gradient of the loss function.
- Pros: Simple and easy to implement, computationally efficient, works well for small datasets.
- Cons: Can be slow to converge, may get stuck in local minima.
### RMSprop
- RMSprop is an adaptive learning rate optimization algorithm that seeks to reduce the aggressive, monotonically decreasing learning rate of Adagrad.
- Pros: Adapts the learning rate for each weight in the network, computationally efficient, works well for non-stationary problems.
- Cons: Can be sensitive to hyperparameter choices.
### Adagrad
- Adagrad is an adaptive learning rate optimization algorithm that adapts the learning rate for each weight in the network based on the frequency of updates.
- Pros: Adapts the learning rate for each weight in the network, requires little configuration, works well for sparse data.
- Cons: Can be computationally expensive, may not work well for non-stationary problems.
# Tests/Combinations and the result
**Since our training data is small in comparison with other projects, I've run every combination with a batch size of 2 in order to get the most accurate gradients as possible. Furthermore, the number of epochs was set to 15 since most of the combinations diverge at that point. In those cases were it didn't converge, I increased the number until it did.**

**Furthermore, I've tried every optimizer with a learning rate ranging from 0.00001 to 0.1.**

The top 3 combinations that were shown to give the smallest training loss while still maintaining a high training accuracy as well as low validation loss while still performing well with regards to validation accuracy were:

## Binary cross-entropy / Adam
Batch size: 2
Number of epochs: 30
Learning rate: 0.001

**Validation loss:** 0.19
**Validation** **accuracy:** 0.94

## Binary Cross / RMSprop
Batch size: 2
Number of epochs: 30
Learning rate: 0.0001

**Validation loss:** 0.41
**Validation** **accuracy:** 0.88

## Binary Cross / RMSprop
Batch size: 2
Number of epochs: 30
Learning rate: 0.001

**Validation loss:** 0.72
**Validation** **accuracy:** 0.86