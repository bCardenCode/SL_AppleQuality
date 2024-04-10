from readApples import readApplesArray
import numpy as np
import math
import csv

class nearestNeighbor():
    def __init__(self, k = 3):
        self.apples = readApplesArray()
        self.predictions = np.array([])
        self.fields = ["Size", "Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", "Acidity"]
        self.numInputs = 7
        self.k = k
        self.quality = "Quality"
    
        
    def EuclideanDistance(self, x, y):
     
        # The sum of the squared 
        # differences of the elements
        sumFeatures = 0; 
        
        for i in range(len(x)):
            # make sure the values are floats
            if type(x[i]) == str:
                x[i] = float(x[i])
            if type(y[i]) == str:
                y[i] = float(y[i])

            sumFeatures += math.pow(x[i]-y[i], 2)
    
        # The square root of the sum
        return math.sqrt(sumFeatures)
    

    # test the k-nn algorithm
    def test(self, trainingLength):
        prediction = np.zeros((len(self.apples)-trainingLength, self.numInputs + 1))

        prediction = self.apples[trainingLength: len(self.apples), :-1]

        # for each apple in the testing set
        for i in range(trainingLength, len(model.apples)):
            apple = self.apples[i]

            # find the k closest apples out of the training set
            closestApples = [100 for i in range(self.k)]
            classification = [0.5 for i in range(self.k)]
            for j in range(trainingLength):
                
                if i == j:
                    continue
                else:
                    distance = self.EuclideanDistance(apple[1:-1], self.apples[j][1:-1])

                    # if distance is one of the closest

                    max_value = max(closestApples)
                    
                    if(distance < max_value):
                        change = closestApples.index(max_value)
                        closestApples[change] = distance
                        classification[change] = self.apples[j][-1]
                
                
        
            averageClassification = sum(classification)/self.k

            
            if averageClassification >= 0.5:
                # print("good")
                prediction[i-trainingLength, -1] = 1
            else:
                # print("bad")
                prediction[i-trainingLength,-1] = 0


        return prediction

        
def calculateAccuracy(answer, prediction):
    correct = 0
    false = 0
    for i in range(len(answer)):
        # print(f"Answer: {answer[i]} Prediction: {prediction[i]}")
        if answer[i] == prediction[i]:
            correct += 1
        else:
            false += 1
    print(f"Correct: {correct} False: {false}")
    print(f"Accuracy: {correct/(correct + false)}")
    return correct/(correct + false)


def saveToCSV(filename, data):
    with open(f"results/{filename}", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


if __name__ == "__main__":
    model = nearestNeighbor()
    trainingLengths = [0.5, 0.7, 0.75, 0.8, 0.9, 0.9]

    # gets the # of rows to train on and test
    trainingLength = 0.01
    trainingRows = int(trainingLength * len(model.apples))
    
    predection = model.test(trainingRows)
    saveToCSV("knnPrediction.csv", predection)

    calculateAccuracy(model.apples[trainingRows: len(model.apples), -1], predection[:, -1])
    print("done")
  
    