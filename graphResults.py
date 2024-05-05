import matplotlib.pyplot as plt
import pandas as pd

def graph_knn_dTree():
    plt.figure(figsize=(10, 6))

    # Read the experiment results from the CSV file
    knn = pd.read_csv('results/knn_experimentResults.csv')
    dTrees = pd.read_csv('results/DTrees_experimentResults.csv')
    # Get the average accuracy and time where the max_depths are the same
    grouped_dTrees = dTrees.groupby('max_depth').mean()
    grouped_knn = knn.groupby('k').mean()

    plt.plot(grouped_dTrees.index, grouped_dTrees['Accuracy'], marker='o', label='Decision Trees (Avg)')
    plt.plot(grouped_knn.index, grouped_knn['Accuracy'], marker='o', label='k-nearest neighbors (Avg)')

    # plt.plot(knn['k'], knn['Accuracy']/knn['Completion Time'], marker='o', label='knn')
    # plt.plot(dTrees['max_depth'], dTrees['Accuracy']/dTrees['Completion Time'], marker='o', label='Decision Trees')
   
    plt.xlabel("k/max_depth")
    plt.ylabel("Completion Time")
    plt.title("K-NN vs Decision Trees Accuracy")
    plt.legend()
    plt.xlim(0, None)
    plt.ylim(0, None)
    plt.show()

def graph_dTree_SciKit():
    plt.figure(figsize=(10, 6))

    # Read the experiment results from the CSV file
    sciKit = pd.read_csv('results/SciKit_experimentResults.csv')
    dTrees = pd.read_csv('results/DTrees_experimentResults.csv')

    # Get the average accuracy and time where the max_depths are the same
    grouped_dTrees = dTrees.groupby('max_depth').mean()
    grouped_sciKit = sciKit.groupby('max_depth').mean()

    plt.plot(grouped_dTrees.index, grouped_dTrees['Completion Time'], marker='o', label='Decision Trees (Avg)')
    plt.plot(grouped_sciKit.index, grouped_sciKit['Completion Time'], marker='o', label='Sci-Kit Learn (Avg)')

    # plt.plot(knn['k'], knn['Accuracy']/knn['Completion Time'], marker='o', label='knn')
    # plt.plot(dTrees['max_depth'], dTrees['Accuracy']/dTrees['Completion Time'], marker='o', label='Decision Trees')
   
    plt.xlabel("k/max_depth")
    plt.ylabel("Completion Time")
    plt.title("Sci-Kit Learn vs Decision Trees Completion Time")
    plt.legend()
    plt.xlim(0, None)
    plt.ylim(0, None)
    plt.show()

if __name__ == "__main__":
    graph_dTree_SciKit()
    print("\n\n FINISH  \n\n")