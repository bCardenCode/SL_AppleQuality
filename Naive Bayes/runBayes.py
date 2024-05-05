
# Sets up reading the input file
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from readApples import readApples

# Models 
from NaiveBayesBins import NaiveBayesBins
from ContinuousBayes import ContinuousBayes

# Stuff
import numpy as np
import matplotlib.pyplot as plt

# Parameters
apples, qualities = readApples()
numBins = 10

# Run Bayes with Bins
# Dict --> Number Bins : Prediction Accuracy
testResults = dict()
for i in range(1, 20):
    binBayes = NaiveBayesBins(apples, i)
    binBayes.train()
    testResults[i] = binBayes.test(binBayes.apples, showOutput = False)
    
print("Bayes with Bins:")    
print(testResults)    
print()

# Run Continuous Bayes
#print("Continuous Bayes:")
#contBayes = ContinuousBayes(apples)
#contBayes.test(contBayes.apples)

"""
# Graph results from Naive Bayes with Bins
plt.bar(list(testResults.keys()), list(testResults.values()))
plt.yticks(np.arange(0.0, 1.0, 0.1))
plt.xticks(np.arange(min(testResults.keys()), max(testResults.keys()) + 1, 1.0))

plt.xlabel("Number of Bins")
plt.ylabel("Prediction Accuracy")
plt.title("Prediction Accuracy vs. Number of Bins")
plt.show()
"""
from sklearn.model_selection import train_test_split
testRatio = 0.75
trainingApples, testingApples, _, _ = train_test_split(apples, qualities, test_size = testRatio)
splitTestResults = dict()
for i in range(1, 20):
    binBayes = NaiveBayesBins(trainingApples, i)
    binBayes.train()
    splitTestResults[i] = binBayes.test(testingApples, showOutput = False)
    
print("Test-Train-Split Bayes with Bins:")    
print(splitTestResults)    
print()


# Graph results from Naive Bayes with Bins
plt.bar(list(splitTestResults.keys()), list(splitTestResults.values()))
plt.yticks(np.arange(0.0, 1.0, 0.1))
plt.xticks(np.arange(min(splitTestResults.keys()), max(splitTestResults.keys()) + 1, 1.0))

plt.xlabel("Number of Bins")
plt.ylabel("Prediction Accuracy")
plt.title("Train-Test Splt Prediction Accuracy vs. Number of Bins")
plt.show()
