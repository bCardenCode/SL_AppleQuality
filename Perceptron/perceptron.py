class Perceptron():
    def __init__(self, apples, learningRate = 0.05, fields = ["Size", "Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", "Acidity"]):
        self.apples = apples
        self.fields = fields
        self.bias = 1
        self.learningRate = learningRate
        self.quality = "Quality"
        self.weights = [0, 0, 0, 0, 0, 0, 0]  
    
    # Converts 'good' or 'bad' to 1 or -1
    def qualityToInt(self, quality):
        if quality == 'good':
            return 1
        else:
            return -1
    
    # Generates a apple quality prediction based on the inputs using current weights    
    def predict(self, apple):
        sum = self.bias
        for i in range(len(self.fields)):   
            sum += float(apple[self.fields[i]]) * self.weights[i]
        
        if sum >= 0:
            return 1
        else:
            return -1

    # Perceptron weight update rule
    def updateWeights(self, diff, apple):
        
        # For all weights
        for i in range(len(self.fields)):
            input = float(apple[self.fields[i]])
            self.weights[i] += self.learningRate * diff * input
    
    # Trains the perceptron on trainingLength apples
    def train(self, apples):
        
        # Trains perceptron
        for apple in apples:
            prediction = self.predict(apple)
            actual = self.qualityToInt(apple[self.quality])
            
            # If wrong, update weights
            if prediction != actual:
                diff = actual - prediction
                self.updateWeights(diff, apple)
                
    # Gets predictions for all apples
    def getPredictions(self, apples):
        return [self.predict(apple) for apple in apples]                  
    
    # Test the perceptron with static weight values. Prints output
    def test(self, apples, showOutput = False):
        correctPredictions = 0
        numApples = len(apples)
        predictions = self.getPredictions(apples)
        
        # Get number of correct predictions
        for i in range(numApples):
            if self.qualityToInt(apples[i][self.quality]) == predictions[i]:
                correctPredictions += 1
        
        # Print results
        if showOutput:
            print("{len(apples)}} Predictions completed.")
            print(f"Score: {correctPredictions} / {numApples} ({correctPredictions / numApples * 100}%)")       
       
        # Return prediction accuracy       
        return round(correctPredictions / numApples, 3)
            

                        