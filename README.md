# MachineLearningProject

## In order to start: 
 
- Create venv 
- Install all the libraries

```

mkdir venv
python3 -m venv ./venv

source venv/bin/activate

pip install --upgrade pip

pip install numpy

pip install tensorflow

pip install keras

pip install scikit-learn

pip install torch torchvision 

```
## Introduction 
After receiving the task to solve, our goal was to build multiple solutions to solve it and compare them. First of all, we compare two different types of really important and famous frameworks for Neural Network: Keras[1] and Pytorch[2]. Then we compare the approach based on Neural Network against a completely different approach in Machine Learning, known as SVM. Finally, we validate, by cross-validation, and test all the models on two different machines to see the performances.

Our first aim is to show our journey from being without any previous knowledge of the frameworks to being able to write two fully working Multi-Layer Perceptrons networks (MLP) with two different shapes. We use two different structures to better understand how they work and how to extend them. The development has been really tough because of the differences between the two frameworks.
We were curious about how to deal with Support Vector Machines (SVM) and so, we choose it too; since we are dealing with a regression task we use a different kind of SVM called Support Vector Regression (SVR) implemented by scikit-learn[3]. The choice of these three models has been done to test all parts of course and to solve in the best way possible the given task. Talking about the task, we have 20 features and 2 target values, each of these are real numbers (they comes from sensors); no other assumption of the data has been done. We have found interesting patterns in the target values, about this it will be discussed later in this report. In order to test our models, we used Hold-out technique to split, initially, the data and then to validate the model; particularly, we used k-fold cross-validation and grid search.
The developed code has been done by Python 3.7. The code is completely automatized; from the main script the user can interact with the models by choosing either if he wants to cross-validate or test one particular instance of a single model, or all of those. All the code is available here. 

