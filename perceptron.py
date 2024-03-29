from readApples import readApples
import numpy as np

class Perceptron():
    def __init__(self, learningRate = 0.25):
        self.apples = readApples()
        self.weights = [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5]
        self.fields = ["Size", "Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", "Acidity"]
        self.bias = 1
        self.numInputs = 7
        self.learningRate = learningRate
        self.quality = "Quality"
    
    # Converts 'good' or 'bad' to 1 or -1
    def qualityToInt(self, quality):
        if quality == 'good':
            return 1
        else:
            return -1
    
    # Generates a apple quality prediction based on the inputs using current weights    
    def predict(self, apple):
        sum = self.bias
        for i in range(self.numInputs):   
            sum += float(apple[self.fields[i]]) * self.weights[i]

        if sum > 0:
            return 1
        else:
            return -1

    # Perceptron update rule. Returns new weight value
    def updateWeight(self, prediction, actual, input, weight):
        if prediction == actual:
            return weight
        else:
            return weight + self.learningRate * (actual - prediction) * float(input)
    
    # Trains the perceptron on traininLength apples
    def train(self, trainingLength = 4000):
        self.weights = [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5]
        for i in range(trainingLength):
            apple = self.apples[i]
            prediction = self.predict(apple)
            actual = self.qualityToInt(apple[self.quality])
            
            for i in range(self.numInputs):
                self.weights[i] = self.updateWeight(prediction, actual, apple[self.fields[i]], self.weights[i])    
    
    # Test the perceptron with static weight values. Prints output
    def test(self, testLength = 4000):
        correctPredictions = 0
        for i in range(testLength):
            apple = self.apples[i]
            appleQuality = self.qualityToInt(apple[self.quality])
            if appleQuality == self.predict(apple):
                correctPredictions += 1
        
        print("4000 Predictions completed")
        print("Score: ", correctPredictions, "/", testLength, "(", correctPredictions / testLength * 100, "%)")        
            
                        