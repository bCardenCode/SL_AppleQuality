import csv
import pandas as pd

"""
Reads apple_quality.csv and converts it to an
array of dictionaries containing apple data
"""
def readApples(fileName = "apple_quality.csv"):
    
    # Stop ID field from being read
    fields = ["Size", "Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", "Acidity", "Quality"]
    
    # Read in apple_quality.csv and convert to list of dictionaries
    csv = pd.read_csv(fileName, usecols = fields)
    dict = csv.to_dict(orient="records")   
    
    # Remove last element (its some random nans for some reason)
    del dict[len(dict) - 1]
    return dict

def readApplesArray(fileName = "apple_quality.csv"):    
    # Read in apple_quality.csv and convert to list of dictionaries
    csv = pd.read_csv(fileName)

    # Read in apple_quality.csv and convert to list of dictionaries
    csv["Quality"] = csv["Quality"].apply(lambda x: 1 if x == "good" else 0)

    return csv.values[:-1]

# Run
if __name__ == "__main__":
    readApples()
