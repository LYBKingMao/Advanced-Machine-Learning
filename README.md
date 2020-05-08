# Advanced-Machine-Learning

## Introduction
This repository contains Level 4's python codes about machine learning which wrote in Jupyter Notebook

## Contents
1. Perceptron from scratch
2. Multi-layer perceptron and its back-propergation from scratch
3. MLP implemented with Keras
4. Word embeddings(self-trained and pre-trained)
5. Convolutional Neuron Network with Keras
6. Recurrent Neuron Network with Keras

## Pre-requisites
1. [Jupyter Notebook](https://jupyter.org/)
2. [Python3](https://www.python.org/downloads/)
3. [Numpy](https://numpy.org/)
4. [Pandas](https://pandas.pydata.org/)
5. [Sklearn](https://scikit-learn.org/stable/)
6. [Keras](https://keras.io/)
7. [Word2vec](https://code.google.com/archive/p/word2vec/)
8. [GloVe](https://nlp.stanford.edu/projects/glove/)
<br>May also contains:
+ [Tensorflow](https://www.tensorflow.org/)\(For training process visualization\)

## Assignments
### Assignment 1
Training Artificial Neural Network (ANN) without third-party libraries like Keras, evaluate model with cross-validation and disscuss
#### Result
Built a MLP with appropriate settings of batches, gradient decent algorithms and achieve ~80% accuracy on testing dataset

### Assignment 2
Using Keras, build regission model and classification model of MLP, CNN and RNN to predict score of product comments, choose a best one and discuss advantages and disadvantages of each architecture
#### Result
Picked CNN as final model, because of unreliable (very easy to overfitting) dataset, using techinogies such as l2 regularization, using confusion matrix to visualize model output and achieve ~75% accuracy on testing dataset
