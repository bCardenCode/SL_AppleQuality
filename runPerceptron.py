from perceptron import Perceptron

perceptron = Perceptron(zeros=True)

print("\nINITIAL TEST----")
perceptron.test()
print()

print("TRAINED TEST----")
perceptron.train()
perceptron.test()
print()


  