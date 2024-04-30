import numpy as np
import readApples

# Define the DecisionTree class
class DecisionTree:
    def __init__(self):
        self.tree = None

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def build_tree(self, X, y):
        # Implement your tree building algorithm here
        pass

    def predict(self, X):
        # Implement your prediction algorithm here
        pass

# Load the apple_quality dataset
def load_data():
    apples = readApples.readApplesArray()
    return apples[:, :-1], apples[:, -1]

# Split the dataset into training and testing sets
def train_test_split(X, y, test_ratio=0.2):

    num_test_rows = int(len(X) * test_ratio)

    test_indices = np.random.choice(len(X), num_test_rows, replace=False)
    train_indices = np.array([i for i in range(len(X)) if i not in test_indices])

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# Main function
if __name__ == "__main__":
    # Load the dataset
    X, y = load_data()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # # Print the shapes of the training and testing sets
    print(f"X_train len: {len(X_train)} {len(X_test)} Y: {len(y_train)} {len(y_test)}")

    # # Create a decision tree classifier
    # classifier = DecisionTree()

    # # Train the decision tree classifier
    # classifier.fit(X_train, y_train)

    # # Make predictions on the testing set
    # y_pred = classifier.predict(X_test)

    # # Evaluate the model
    # accuracy = np.mean(y_pred == y_test)
    # print("Accuracy:", accuracy)

