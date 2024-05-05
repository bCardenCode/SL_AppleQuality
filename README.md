# SL_AppleQuality
This is the repository for the Supervised Learning project to predict Apple Quality based on a dataset provided by Kaggle. Code was written by Brady Carden and Gabe Owen for our Machine Learning Fundamentals course at OU.

To run k nearest neighbors:

- at the bottom of the file you can adjust the values following values
    - ks for values of k to train on
    - training length for the split of training data
    - num_runs for how many times each k in ks will run
    - save file for the directory of the output as a csv

- once the values are adjusted to your liking run 'python3 nearestNeighbor.py'


To run Decision Trees:

- at the bottom of the file you can adjust the values following values
    - run_depths for multiple models to be created with the max_depth of each int in run_depths 
    - num_runs for how many times each k in ks will run
    - save file for the directory of the output as a csv

- once the values are adjusted to your liking run 'python3 DecisionTrees.py'
- inside of this file also contains the implementation of Sci-kit Learns Decision Trees for comparison