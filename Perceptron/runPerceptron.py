# Sets up reading the input file
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from readApples import readApples

# Model and stuffs
from perceptron import Perceptron
import matplotlib.pyplot as plt
import numpy as np

# Set up data
apples, qualities = readApples()
perceptron = Perceptron(apples)

# Testing
testResults = []
trainingIterations = 10
for i in range(trainingIterations):
    testResults.append(perceptron.test(apples))
    perceptron.train(apples)
    
print(testResults)    
 
""" 
# Graph results from Naive Bayes with Bins
plt.plot(range(1, trainingIterations + 1), testResults)
plt.yticks(np.arange(0.0, 1.0, 0.1))
plt.xticks(np.arange(1, trainingIterations + 1, 1.0))

plt.xlabel("Training Iterations")
plt.ylabel("Prediction Accuracy")
plt.title("Prediction Accuracy vs. Training Iterations")
plt.show()  
"""

from sklearn.model_selection import train_test_split
testRatio = 0.75
trainingApples, testingApples, _, _ = train_test_split(apples, qualities, test_size = testRatio)
perceptron = Perceptron(testingApples)
splitTestResults = []
for i in range(trainingIterations):
    splitTestResults.append(perceptron.test(testingApples))
    perceptron.train(trainingApples)
    
print(splitTestResults)    

# Graph results from Naive Bayes with Bins
plt.plot(range(1, trainingIterations + 1), splitTestResults)
plt.yticks(np.arange(0.0, 1.0, 0.1))
plt.xticks(np.arange(1, trainingIterations + 1, 1.0))

plt.xlabel("Training Iterations")
plt.ylabel("Prediction Accuracy")
plt.title("Test-Traing Split: Prediction Accuracy vs. Training Iterations")
plt.show()  
