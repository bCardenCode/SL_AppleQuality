from readApples import readApplesArray
import numpy as np
import math

class nearestNeighbor():
    def __init__(self, k = 3):
        self.apples = readApplesArray()
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

        # gets the # of rows to train on and test
        trainingRows = int(trainingLength * len(model.apples))

        # for each apple in the testing set
        for i in range(trainingRows, len(model.apples)):
            apple = self.apples[i]

            # find the k closest apples out of the training set
            closestApples = [100 for i in range(self.k)]
            classification = [0.5 for i in range(self.k)]
            for j in range(trainingRows):
                
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
                print("good")
                # apple[-1] = 1
            else:
                print("bad")
                # apple[-1] = 0

        # for i in range(trainingLength):
        #     apple = self.apples[i]

        #     # find the k closest apples
        #     closestApples = [100 for i in range(self.k)]
        #     for j in range(trainingLength):
        #         if i == j:
        #             continue
        #         else:
        #             distance = self.EuclideanDistance(apple[1:-1], self.apples[j][1:-1])

        #             # if distance is one of the closest
        #             min_value = max(closestApples)
        #             if(distance < min_value):
        #                 closestApples.index(min_value)  
        
        #     sum = 0
        #     for i in range(self.k):
        #         sum += closestApples[i]
        #     sum = sum / self.k

        #     if sum >= 0.5:
        #         print("good")
        #         apple[-1] = 1
        #     else:
        #         print("bad")
        #         apple[-1] = 0
        


if __name__ == "__main__":
    model = nearestNeighbor()
    trainingLengths = [0.5, 0.7, 0.75, 0.8, 0.9, 0.9]

    # [1:-1] removes the id and the quality from the apple
    
    model.test(trainingLength = 0.75)
    print("done")
  

    # for length in trainingLengths:
    #     trainingRows = int(length * len(model.apples))
    #     testingRows = len(model.apples) - trainingRows
        
    #     print(f"\nTRAINED on ({length}) = {length} APPLES TEST {testingRows}")
    #     model.train(trainingRows)
    #     model.test()
    #     print()
    