from perceptron import Perceptron

perceptron = Perceptron()
trainingLengths = [100, 500, 1000, 2000, 3000, 4000]

print("\nINITIAL TEST----")
perceptron.test()
print()
"""
for length in trainingLengths:
    print("TRAINED on", length, "APPLES TEST----")
    perceptron.train(length)
    perceptron.test()
    print()
"""

for _ in range(4):
    perceptron.train(resetWeights=False)
    perceptron.test()   