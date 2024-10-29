# Supervised-Learning-with-Artificial-Neural-Networks-for-Regression-Analysis
# Overview
This project focuses on developing and analyzing Artificial Neural Networks (ANNs) as regressors for various regression problems. The study investigates two key configurations of ANNs: a linear regressor without a hidden layer and another with a single hidden layer. It emphasizes the use of the backpropagation algorithm for learning weights and analyzes how different network parameters impact model performance.

# Key Objectives
Implement and train ANN models for regression tasks using a given dataset.
Evaluate the performance of different ANN configurations, focusing on the effects of hidden layers, neuron count, learning rate, weight initialization, and epoch count.
Optimize the models for lower loss and better accuracy using a systematic approach, including normalization, Xavier initialization, and stochastic gradient descent.
# Implementation Details
# Model Architecture
# Linear Regressor (No Hidden Layer):
This model consists of a single neuron with a linear activation function. It serves as a baseline model, behaving like simple linear regression.
The network uses normalization for input data preprocessing and applies backpropagation for weight optimization.
# ANN with a Single Hidden Layer:
This model features one hidden layer using a sigmoid activation function, followed by an output layer with a linear activation function.
A grid search is employed to determine the optimal number of neurons in the hidden layer, ranging from 2 to 32.
# Training and Optimization
Normalization: Input data is normalized to ensure efficient learning.
Xavier Initialization: Weights are initialized to maintain the gradient scale across layers, supporting effective backpropagation.
Loss Function: The sum of squared errors (SSE) is used to evaluate model performance.
Stochastic Gradient Descent (SGD): Used for weight updates, with trial-and-error employed to determine the optimal learning rate and epoch count.
The model parameters are tuned iteratively, with learning rates set at 0.012 and 0.01 for the two configurations, respectively.
Results
# Linear Regressor (No Hidden Layer)

Optimal Parameters: Learning rate = 0.012, epochs = 100.
Training Loss: 0.19, Test Loss: 11.21.
The model serves as a baseline but struggles with complex data due to the lack of non-linearity.
# ANN with Single Hidden Layer

Optimal Parameters: 20 neurons, learning rate = 0.01, epochs = 10,000.
Training Loss: 0.035, Test Loss: 2.12.
The hidden layer significantly improves model performance, reducing both training and test losses.
# Comparison Across Models

Increasing the number of hidden units generally improves performance, but excessive complexity can lead to overfitting, as seen in models with over 20 neurons.
# Challenges
Optimal Parameter Selection: Determining the ideal number of neurons, learning rate, and epochs through trial and error.
Normalization Impact: The model performed poorly without normalization, highlighting its importance in achieving better accuracy and faster convergence.
Overfitting: Balancing model complexity to prevent overfitting, especially with higher neuron counts in the hidden layer.
# Conclusion
The study demonstrates that ANN configurations with hidden layers significantly enhance regression performance compared to simple linear models. By effectively tuning network parameters and employing systematic techniques like normalization and Xavier initialization, the models achieved optimal performance. The project highlights the importance of proper architecture design and parameter tuning in supervised learning applications.

# Future Improvements
Implement additional ANN configurations, such as models with multiple hidden layers, to explore further performance gains.
Use more advanced optimization algorithms like Adam or RMSprop to potentially achieve faster convergence and lower loss.
Apply the models to more complex datasets to evaluate generalizability and robustness.
