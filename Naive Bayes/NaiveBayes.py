import os
import sys
import inspect
import numpy as np

# This is used to import from a parent folder
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from readApples import readApples

class NaiveBayes():
    def __init__(self, numApples, numBins):
        self.numApples = numApples # Number of apples to be trained on
        self.numBins = numBins
        self.apples = readApples()
        self.fields = ["Size", "Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", "Acidity"]
        self.goodApples = 0
        self.badApples = 0
        
        # Maxes and Mins for each input category
        self.inputRanges = {
            "Size": (-7.15, 6.41),
            "Weight": (-7.15, 5.79),
            "Sweetness": (-6.89, 6.37),
            "Crunchiness": (-6.06, 7.62),
            "Juiciness": (-5.96, 7.36),
            "Ripeness": (-5.86, 7.24), 
            "Acidity": (-7.01, 7.4)
        }
        
        # Creates the bins for each input
        self.inputBins = dict()        
        for field in self.fields:
            range = self.inputRanges[field]
            self.inputBins[field] = np.linspace(range[0], range[1], self.numBins)
                
        self.goodProbs = dict.fromkeys(self.fields, np.zeros(numBins))
        self.badProbs = dict.fromkeys(self.fields, np.zeros(numBins)) 
    
    def getGoodGetBad(self):
        self.goodApples = 0
        for i in range(len(self.apples)):
            if self.apples[i]["Quality"] == "good":
                self.goodApples += 1
        self.badApples = len(self.apples) - self.goodApples
    
    # Gets probabilities that the apple is good given that an input is positive    
    def getProbs(self):
        for apple in self.apples:
            for field in self.fields:
                inputBin = np.digitize(apple[field], self.inputBins[field]) - 1
                if apple["Quality"] == "good":
                    self.goodProbs[field][inputBin] += 1
                else:
                    self.badProbs[field][inputBin] += 1    
        
        for field in self.fields:
            for bin in range(self.numBins):
                self.goodProbs[field][bin] /= self.goodApples
                self.badProbs[field][bin] /= self.badApples
    
    def predict(self, apple):
        goodProb = 1
        badProb = 1
        for field in self.fields:
            bin = np.digitize(apple[field], self.inputBins[field]) - 1
            goodProb *= self.goodProbs[field][bin]
            badProb *= self.badProbs[field][bin]
            
        if goodProb >= badProb:
            return "good"    
        else:
            return "bad"    
            
    def test(self, testLength = 4000):
        correctPredictions = 0
        for apple in self.apples:
            if apple["Quality"] == self.predict(apple):
                correctPredictions += 1
        print("4000 Predictions completed.")
        print(f"Score: {correctPredictions} / {testLength} ({correctPredictions / testLength * 100}%)")            
            
                                