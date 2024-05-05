import numpy as np

class ContinuousBayes():
    def __init__(self, apples, fields = ["Size", "Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", "Acidity"] ):
        self.numApples = len(apples) 
        self.apples = apples
        self.fields = fields
        
        # Dicts containing values for each field for good apples and bad apples
        # Used to find stats as seen below
        self.goodAppleFieldValues = dict.fromkeys(self.fields, [])
        self.badAppleFieldValues = dict.fromkeys(self.fields, [])
        for apple in self.apples:
            if apple["Quality"] == "good":
                # Append apple's fields to goodAppleFieldValues
                for field in self.fields:
                    self.goodAppleFieldValues[field].append(float(apple[field]))
            else:
                # Append apple's field values to badAppleFieldValues
                for field in self.fields:    
                    self.badAppleFieldValues[field].append(float(apple[field]))
        
        # Dicts containing average values for each field for good apples and bad apples
        self.goodFieldAvgs = {field: sum(self.goodAppleFieldValues[field]) / len(self.goodAppleFieldValues[field]) for field in self.fields}     
        self.badFieldAvgs = {field: sum(self.badAppleFieldValues[field]) / len(self.badAppleFieldValues[field]) for field in self.fields}   
        
        # Dicts containing standard deviation values for each field for good apples and bad apples
        self.goodFieldSTDs = {field: np.std(self.goodAppleFieldValues[field]) for field in self.fields}
        self.badFieldSTDs = {field: np.std(self.badAppleFieldValues[field]) for field in self.fields}
        
        # Dicts containing variance values for each field for good apples and bad apples
        self.goodFieldVariances = {field: np.var(self.goodAppleFieldValues[field]) for field in self.fields}
        self.badFieldVariances = {field: np.var(self.badAppleFieldValues[field]) for field in self.fields}
    
      # Predicts using continuous naive bayes prediction
    def predict(self, apple):
        # Names in this method are pretty bad but it just calculates the formula shown in the paper
        goodProb = 1
        badProb = 1
        for field in self.fields:
            goodFrac1 = 1 / (np.sqrt(2 * np.pi) * self.goodFieldSTDs[field])
            goodFrac2 = -1 * (np.power(float(apple[field]) - self.goodFieldAvgs[field], 2)) / (2 * self.goodFieldVariances[field])
            goodE = np.exp(goodFrac2)
            
            goodProbX = goodFrac1 * goodE
            goodProb *= goodProbX
            
            badFrac1 = 1 / (np.sqrt(2 * np.pi) * self.badFieldSTDs[field])
            badFrac2 = -1 * (np.power(float(apple[field]) - self.badFieldAvgs[field], 2)) / (2 * self.badFieldVariances[field])
            badE = np.exp(badFrac2)
            
            badProbX = badFrac1 * badE
            badProb *= badProbX
        
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
        return correctPredictions / numApples     

