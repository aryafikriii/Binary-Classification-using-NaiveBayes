# Binary Classification using Naive Bayes

This project focuses on implementing a Naive Bayes classifier for binary classification using the provided dataset in the file "traintest.xlsx". The dataset consists of two sheets: "traindantest" containing the training and testing data, and "train" containing 296 rows of labeled data for model training. The dataset contains input features (?1 to ?3) represented by integer values within a specific range, and a binary output class (0 or 1).

## Method Selection: Naive Bayes

For this project, the chosen method is Naive Bayes. Naive Bayes is a probabilistic classifier that uses Bayes' theorem with the assumption of independence between features. It calculates the probability of each class given the input features and selects the class with the highest probability.

## Implementation

The following processes already implemented in the program:

1. Data Reading: Read the training and testing data from the "traintest.xlsx" file.

2. Model Training: Use the training data to train the Naive Bayes model, considering the independence assumption between features.

3. Model Saving: Save the trained model to be used for testing and future predictions.

4. Model Testing: Apply the trained model on the testing data to predict the class labels.

5. Model Evaluation: Evaluate the performance of the model by comparing the predicted class labels with the actual class labels.

6. Output Saving: Save the output of the testing phase, including the predicted class labels, to a file.

Note: The training, testing, and evaluation processes implemented without using any external libraries, building each step from scratch.

## Contact

If you have any questions or suggestions regarding this project, please feel free to contact me. You can reach me at [your_email@example.com].

Thank you for your interest in the Binary Classification using Naive Bayes project. Happy coding!
