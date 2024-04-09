from readApples import readApples
import numpy as np
import math

class nearestNeighbor():
    def __init__(self, k = 3):
        self.apples = readApples()
        self.weights = [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5]
        self.fields = ["Size", "Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", "Acidity"]
        self.bias = 1
        self.numInputs = 7
        self.k = k
        self.quality = "Quality"
    
    # Converts 'good' or 'bad' to 1 or -1
    def qualityToInt(self, quality):
        if quality == 'good':
            return 1
        else:
            return -1
        
    def EuclideanDistance(self, x, y):
     
        # The sum of the squared 
        # differences of the elements
        sumFeatures = 0; 
        
        for key in x.keys():
            sumFeatures += math.pow(x[key]-y[key], 2)
    
        # The square root of the sum
        return math.sqrt(sumFeatures)
    

    # Trains the perceptron on traininLength apples
    def train(self, trainingLength):
        
        for i in range(trainingLength):
            apple = self.apples[i]
            prediction = self.predict(apple)
            actual = self.qualityToInt(apple[self.quality])
            
            for i in range(self.numInputs):
                self.weights[i] = self.updateWeight(prediction, actual, apple[self.fields[i]], self.weights[i])    
    
    # Test the perceptron with static weight values. Prints output
    def test(self, testLength):
        correctPredictions = 0
        for i in range(testLength):
            apple = self.apples[i]
            appleQuality = self.qualityToInt(apple[self.quality])
            if appleQuality == self.predict(apple):
                correctPredictions += 1
        
        print("Score: ", correctPredictions, "/", testLength, "(", correctPredictions / testLength * 100, "%)")        
        
if __name__ == "__main__":
    model = nearestNeighbor()
    trainingLengths = [0.5, 0.7, 0.75, 0.8, 0.9, 0.9]

    print(type(model.apples[0]))
    print(model.apples[0][:-1])
    print(model.apples[1][:-1])
    print("euclidean distance between 2:", model.EuclideanDistance(model.apples[0][:-1], model.apples[1][:-1]))

    # for length in trainingLengths:
    #     trainingRows = int(length * len(model.apples))
    #     testingRows = len(model.apples) - trainingRows
        
    #     print(f"\nTRAINED on ({length}) = {length} APPLES TEST {testingRows}")
    #     model.train(trainingRows)
    #     model.test()
    #     print()
    