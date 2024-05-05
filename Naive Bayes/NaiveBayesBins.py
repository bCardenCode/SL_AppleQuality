import numpy as np

class NaiveBayesBins():
    def __init__(self, apples, numBins, fields = ["Size", "Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", "Acidity"] ):
        self.numApples = len(apples) # Number of apples to be trained on
        self.numBins = numBins
        self.apples = apples
        self.fields = fields
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
        
        # Dicts containing probabilites for each field and their bins for bin prediction        
        self.goodProbs = dict.fromkeys(self.fields, np.zeros(numBins))
        self.badProbs = dict.fromkeys(self.fields, np.zeros(numBins)) 
        
    # Gets the number of good apples and number of bad apples
    def getNumGoodNumBad(self):
        # Get number of good Apples
        self.goodApples = 0
        for apple in self.apples:
            if apple['Quality'] == 'good':
                self.goodApples += 1
        
        # Calculate number of bad apples        
        self.badApples = self.numApples - self.goodApples
    
    # Gets probabilities that the apple is good given that an input is positive    
    def getBinProbs(self):
        for apple in self.apples:
            # Gets the number of occurences of each field's bins for good and bad apples
            for field in self.fields:
                inputBin = np.digitize(apple[field], self.inputBins[field]) - 1
                if apple["Quality"] == "good":
                    self.goodProbs[field][inputBin] += 1
                else:
                    self.badProbs[field][inputBin] += 1    
        
        # Divides each bin by the num of good or bad apples to make these a percent value
        for field in self.fields:
            for bin in range(self.numBins):
                self.goodProbs[field][bin] /= self.goodApples
                self.badProbs[field][bin] /= self.badApples
    
    # Not really training, but gets the model ready to predict
    def train(self):
        self.getNumGoodNumBad()
        self.getBinProbs()
    
    # Predicts apple quality using bin probs
    def predict(self, apple):
        
        # init likelihoods for good and bad quality
        goodProb = 1
        badProb = 1
        
        for field in self.fields:
            # Get bin for value of each field and update probabilities
            bin = np.digitize(apple[field], self.inputBins[field]) - 1
            goodProb *= self.goodProbs[field][bin]
            badProb *= self.badProbs[field][bin]
            
        # Determine which is more likely    
        if goodProb >= badProb:
            return "good"    
        else:
            return "bad"    
        
    # Returns an array of predictions for the provided set of apples        
    def getPredictions(self, apples):
        return [self.predict(apple) for apple in apples]                
  
    # Returns the percent accuracy        
    def test(self, apples, showOutput = True):
        correctPredictions = 0
        numApples = len(apples)
        predictions = self.getPredictions(apples)
        
        # Get number of correct predictions
        for i in range(numApples):
            if apples[i]["Quality"] == predictions[i]:
                correctPredictions += 1
        
        # Print results        
        if showOutput:        
            print("4000 Predictions completed.")
            print(f"Score: {correctPredictions} / {numApples} ({correctPredictions / numApples * 100}%)")      
            
        # Return percent accuracy    
        return round(correctPredictions / numApples, 3)  
          

            
                                