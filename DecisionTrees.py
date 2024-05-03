import numpy as np
import readApples

class Node:
    def __init__(self, data=None, children=None, split_on = None, pred_class=None, is_leaf=False, depth=0):
        self.data = data
        self.children = children
        self.split_on = split_on
        self.pred_class = pred_class
        self.is_leaf = is_leaf
        self.depth = depth
        

        
# Define the DecisionTree class
class DecisionTree:
    def __init__(self, fullData, split_on, max_depth=5):
        self.root = Node(data=fullData, split_on=split_on)
        self.max_depth = max_depth
        self.current_depth = 0
        self.fullData = fullData
        self.num_features = len(fullData[0]) - 1
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    # start at the root node and recursively split the data
    def fit(self):
        self.build_tree(self.fullData, self.current_depth)

    def build_tree(self, data, depth):

        # find best split
        best_gain, best_question = self.find_split(data)
        print(f"Best Gain: {best_gain}, Best Question: {best_question}")
        return 0
    
        # if no split is found, make the node a leaf node

        # split the data true/false on the best split

        # build tree for true

        # build tree for false

        # return node




    def find_split(self, data):

        best_gain = 0  
        best_question = None 
        current_uncertainty = self.gini_impurity(data)

        # for each feature
        for feature_index in range(self.num_features): 

            # get the unique values of the selected feature
            unique_selected_feature_apples = set([apple[feature_index] for apple in data]) 

            for apple in unique_selected_feature_apples: 

                # save the question as a tuple
                question = (feature_index, apple)

                # split the data based of question
                true_rows, false_rows = self.question_split(data, question)


                # skip if the split is not valid
                if len(true_rows) == 0 or len(false_rows) == 0:
                    print(f"true_rows = {len(true_rows)} and false_rows = {len(false_rows)}. Skipping...")
                    continue

                # Calculate the information gain from this split
                percent_true = len(true_rows) / len(data)
                info_gain = current_uncertainty - percent_true * self.gini_impurity(true_rows) - (1 - percent_true) * self.gini_impurity(false_rows)


                # update the best gain and question if better gain
                if info_gain >= best_gain:
                    best_gain, best_question = info_gain, question

        return best_gain, best_question


    # Split the data based on the question (feature index, value)
    def question_split(self, data, question):
        feature_index, value = question

        true_rows = []
        false_rows = []
        for apple in data:
            if apple[feature_index] >= value:
                true_rows.append(apple)
            else:
                false_rows.append(apple)

        return np.array(true_rows), np.array(false_rows)

    # Gini impurity fucntion
    def gini_impurity(self, data):


        # get the count of each label in the data
        counts = [np.sum(data[:, -1] == 0), np.sum(data[:, -1] == 1)]
        impurity = 1 

        # calculate the gini impurity
        for label in counts:
            label_prob = label / len(data)
            impurity -= label_prob ** 2

        return impurity

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
    classifier = DecisionTree(fullData=data, split_on=0, max_depth=5)
    classifier.train_test_split()

    # testing prints
    # print("Length of X_train:", len(classifier.X_train)) 
    # print("Length of y_train:", len(classifier.y_train)) 
    # print("Length of X_test:", len(classifier.X_test)) 
    # print("Length of y_test:", len(classifier.y_test))


    # # Train the decision tree classifier
    classifier.fit()

    # # Make predictions on the testing set
    # y_pred = classifier.predict(X_test)

    # # Evaluate the model
    # accuracy = np.mean(y_pred == y_test)
    # print("Accuracy:", accuracy)

