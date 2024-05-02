import numpy as np
import readApples

class Node:
    def __init__(self, data=None, children=None, split_on = None, pred_class=None, is_leaf=False):
        self.data = data
        self.children = children
        self.split_on = split_on
        self.pred_class = pred_class
        self.is_leaf = is_leaf
        

        
# Define the DecisionTree class
class DecisionTree:
    def __init__(self, fullData, split_on):
        self.root = Node(data=fullData, split_on=split_on )
        self.fullData = fullData
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def build_tree(self, X, y):
        # Implement your tree building algorithm here
        pass

    def predict(self, X):
        # Implement your prediction algorithm here
        pass

    # Split the dataset into training and testing sets
    def train_test_split(self, test_ratio=0.2):

        X = self.fullData[:, :-1]
        y = self.fullData[:, -1]
        num_test_rows = int(len(X) * test_ratio)

        test_indices = np.random.choice(len(X), num_test_rows, replace=False)
        train_indices = np.array([i for i in range(len(X)) if i not in test_indices])

        self.X_train = X[train_indices]
        self.y_train = y[train_indices]
        self.X_test = X[test_indices]
        self.y_test = y[test_indices]

        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# Main function
if __name__ == "__main__":
    # Load the dataset
    data = readApples.readApplesArray()


    # Create a decision tree classifier
    classifier = DecisionTree(fullData=data, split_on=0)
    classifier.train_test_split()

    # testing prints
    print("Length of X_train:", len(classifier.X_train)) 
    print("Length of y_train:", len(classifier.y_train)) 
    print("Length of X_test:", len(classifier.X_test)) 
    print("Length of y_test:", len(classifier.y_test))

    # # Train the decision tree classifier
    # classifier.fit(X_train, y_train)

    # # Make predictions on the testing set
    # y_pred = classifier.predict(X_test)

    # # Evaluate the model
    # accuracy = np.mean(y_pred == y_test)
    # print("Accuracy:", accuracy)

