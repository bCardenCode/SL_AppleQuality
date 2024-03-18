from NaiveBayes import NaiveBayes

bayes = NaiveBayes(4000, 10)

print("UNTRAINED TEST----")
bayes.test()
print()

print("TRAINED TEST----")
bayes.getGoodGetBad()
bayes.getProbs()
bayes.test()