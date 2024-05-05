from readApples import readApplesArray
import numpy as np
import math
import csv
import time
import pandas as pd
import os
import matplotlib.pyplot as plt

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

            # find the nearest neighbors out of all the training apples
            for j in range(trainingLength):
                    
                
                distance = self.EuclideanDistance(apple[1:-1], self.apples[j][1:-1])

                # get the furthest saved neighbor
                max_value = max(closestApples)
                
                # if the new distance is less than the furthest saved neighbor
                if(distance < max_value):
                    # then save the new distance and classification
                    change = closestApples.index(max_value)
                    closestApples[change] = distance
                    classification[change] = self.apples[j][-1]
                
                
            # calculate the average classification of the k closest apples
            averageClassification = sum(classification)/self.k

            # decide if the apple is good or bad
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

    
def runAndSaveExpiriment(model, trainingLengths):
    data = []
    for length in trainingLengths:
        start_time = time.time()
        
        trainingRows = int(length * len(model.apples))

        predection = model.test(trainingRows)

        # saveToCSV(f"k({model.k})nnPrediction{length}.csv", predection)

        accuracy = calculateAccuracy(model.apples[trainingRows: len(model.apples), -1], predection[:, -1])

        completionTime = time.time() - start_time

        # Save model.k, length, accuracy, and time to a CSV file
        data.append([model.k, length, accuracy, completionTime])

        print("done")
    
    # save data from multiple runs to a CSV file
    with open(f"results/experimentResults.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def graphExpirement(fileName):
    # Read the experiment results from the CSV file
    data = pd.read_csv(fileName)



    accuracies = data['Accuracy']
    trainingLengths = data['k']
    plt.plot(trainingLengths, accuracies)



    # Extract the training lengths and accuracies
    # accuracies = data['Accuracy']
    # trainingLengths = data['Training Length']
    

    # Plot the accuracy vs training length
    # plt.plot(trainingLengths, accuracies)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs k")
    plt.show()



if __name__ == "__main__":
    saveFile = "results/experimentResults.csv"

    # Clear the contents of saveFile
    with open(saveFile, 'w') as file:
        file.truncate(0)
        writer = csv.writer(file)
        writer.writerow(["k", "Training Length", "Accuracy", "Completion Time"])



    # # run multiple expiriments
    for _ in range(1):
         ks = [1, 3, 5, 10, 15, 20]
         for k in ks:
             model = nearestNeighbor(k=k)
             trainingLengths = [0.75]
             runAndSaveExpiriment(model, trainingLengths)

    # # run one expiriment
    # model = nearestNeighbor(k=1)
    # runAndSaveExpiriment(model, [0.75])


    graphExpirement(fileName=saveFile)

    print("\n\n FINISH  \n\n")
  
    