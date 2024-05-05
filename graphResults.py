import matplotlib.pyplot as plt
import pandas as pd

def graphExpirement():

    # Read the experiment results from the CSV file
    knn = pd.read_csv('results/knn_experimentResults.csv')
    dTrees = pd.read_csv('results/DTrees_experimentResults.csv')



    knn = knn.sort_values('Completion Time')

    accuracies = knn['Accuracy']
    time = knn['Completion Time']
    plt.plot(time, accuracies)

    accuracies = dTrees['Accuracy']
    time = dTrees['Completion Time']
    plt.plot(time, accuracies)



    plt.legend(["knn", "Decision Trees"])
    plt.xlabel("Time")
    plt.ylabel("Accuracy")
    plt.title("knn vs Decision Trees")
    plt.show()

if __name__ == "__main__":
    graphExpirement()
    print("\n\n FINISH  \n\n")