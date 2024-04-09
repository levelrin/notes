## General Flow of Making Predictions

Data -> Model -> Prediction

---

Two types of data: X and Y.
- X is the training dataset used to make predictions.<br>
  AKA. Attributes or Feature values<br>
  Imagine we have a table. Each column represents an attribute (feature).<br>
  Ex: House Info.
- Y is the training dataset that represents predictions.<br>
  AKA. Targets or labels<br>
  Ex: Corresponding house prices.

---

Two phases in the model step:
1. Create a model object.
2. Learn from the data (fit to the data).

---

To make a model, we can do the following:
1. Apply scaling to the dataset.
2. Select the algorithm, such as a linear regression.

---

Scaling is a process of transforming your feature values to fit within a specific range. Ex: transforming the dataset unit-free.

We might want to apply scaling before applying the algorithm for various reasons.

For example, let's say we want to predict the price of a house using the number of shops nearby and the number of years since it was built.

The number of shops nearby might almost always be larger than the number of years since the house was built.

For that reason, the prediction might be biased towards the number of shops nearby.

To equalize the influence of each feature, we may want to apply scaling.

Additionally, some algorithms may require scaling before using them anyway.

## Load dataset for education

```python
from sklearn.datasets import load_diabetes
import pandas as pd


def main():
    # Load the Diabetes dataset
    # https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset
    dataset = load_diabetes()

    # Create a Pandas DataFrame
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

    # Add the target variable 'Disease Progression' to the DataFrame
    df['Disease Progression'] = dataset.target

    # Display the entire DataFrame without truncation.
    # Note that the dataset does not use the actual values.
    # For example, you will see one of the values in the 'age' column is -0.001882.
    # Instead, they are preprocessed feature values to help algorithms perform better.
    # In other words, the dataset has undergone feature scaling or standardization.
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)


main()

```

## Make predictions

```python
from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor


def main():
    # Load the dataset in x, y format.
    x, y = load_diabetes(return_X_y=True)

    # Create KNeighborsRegressor model.
    model = KNeighborsRegressor()

    # Learn from the dataset.
    model.fit(x, y)

    # Make predictions.
    predictions = model.predict(x)

    # The number of predictions equals the number of rows in the 'x' array.
    print(predictions)


main()

```

## Show the scatter plot to check the performance of the model

```python
from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pylab as plot


def main():
    x, y = load_diabetes(return_X_y=True)
    model = KNeighborsRegressor()
    model.fit(x, y)
    predictions = model.predict(x)

    # Create a scatter plot that shows the relationship between the predicted values and the actual target values.
    plot.scatter(predictions, y)

    # A new window will pop up.
    plot.show()


main()

```

## Pipeline

A pipeline is used to chain the processes of the model.

```python
from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pylab as plot


def main():
    x, y = load_diabetes(return_X_y=True)

    # Create a pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", KNeighborsRegressor())
    ])

    # Pipeline can do what the model can.
    pipeline.fit(x, y)
    predictions = pipeline.predict(x)

    plot.scatter(predictions, y)
    plot.show()


main()

```

## GridSearchCV

It's for finding the optimal parameter values from a given set of parameters in the model.

In other words, it's for hyperparameter tuning.

Hyperparameter means it's a configuration data for the model.

And it's unrelated to the training data we feed into the model.

---

It's also used to perform cross-validation.

It means we split the dataset into multiple subsets and use them to test the model's performance.

For example, let's say we have the following dataset:
| Years of Experience | Salary |
| ------------------- | ------ |
| 1                   | 10     |
| 2                   | 18     |
| 3                   | 25     |
| 4                   | 44     |

If we use the above dataset for both training and predicting (testing), the model would look too good because the model was using the answers during training.

For that reason, we want to have separate datasets, one for training and another for prediction (testing).

The trick is to divide the dataset into multiple subsets like this:

Subset1
| Years of Experience | Salary |
| ------------------- | ------ |
| 1                   | 10     |
| 2                   | 18     |

Subset2
| Years of Experience | Salary |
| ------------------- | ------ |
| 3                   | 25     |
| 4                   | 44     |

And then, we use subset1 for training and subset2 for testing.

---

Here is a sample code:

```python
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plot


def main():
    x, y = load_diabetes(return_X_y=True)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", KNeighborsRegressor())
    ])

    # Our model is getting more complicated :)
    model = GridSearchCV(
        estimator=pipeline,
        # Hyperparameters.
        # We want to find their best values.
        param_grid={
            # You can check the available hyperparameters by running "print(pipeline.get_params())"
            # It will find the best value of this hyperparameter from 1 to 10.
            'regressor__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        },
        # It's the number of cross-validation folds (subsets of the dataset).
        # In other words, it's the number of divisions we make on the dataset.
        # Higher value would make the model "better" in general.
        # However, it will increase the computational cost and reduce training data size.
        # A small training dataset caused by the high cv value might make the model "worse."
        cv=3
    )

    # GridSearchCV object can also perform fit.
    model.fit(x, y)
    
    # We can check the information of the search results.
    results = model.cv_results_
    df = pd.DataFrame(results)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

    # GridSearchCV object can also perform predict with the best hyperparameters.
    predictions = model.predict(x)
    
    plot.scatter(predictions, y)
    plot.show()


main()

```

---

We can configure metrics for `GridSearchCV`.
It will select the best hyperparameters based on the metric we specified.

Here is an example:
```python
model = GridSearchCV(
    # We can use this parameter to specify the metrics that we want to use.
    scoring={
        # make_scorer, precision_score, and recall_score are provided by scikit-learn.
        # Make sure you apply metrics compatible with the estimator (algorithm).
        'precision': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score)
    },

    # We must use this parameter if we use the 'scoring' parameter.
    # The GridSearchCV object will use this metric to generate a rank and pick the best hyperparameter.
    # You may wonder what's the point of specifying multiple metrics then.
    # Well, we can see the 'model.cv_results_' to check how the algorithm performs under various metrics.
    # In other words, it's just for our information.
    refit='precision',

    # The rest of the parameters are irrelevant to the topic.
    estimator=LogisticRegression(max_iter=1000),
    param_gried={
        'class_weight': [
            {0: 1, 1: v} for v in range(1, 4)
        ]
    },
    cv=4,
    n_jobs=-1
)
```

---

`GridSearchCV` is an example of meta-estimators.

A `meta-estimator` is an estimator that takes another estimator as a parameter.

## Neural Networks

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# All data for training.
x = [0.5, 1, 1.5, 2, 2.5, 4, 4.5, 5, 5.5, 6, 7.5, 8, 8.5, 9, 9.5]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# Artifical Neural Network (ANN)
model = MLPClassifier(
	# (2,) means 1 hidden layer with 2 nodes.
	# (3, 4) means 2 hidden layers. First layer has 3 nodes and second layer has 4 nodes.
	# (5, 6, 7) means 3 hidden layers. First layer has 5 nodes, second layer has 6 nodes, and third layer has 7 nodes.
	hidden_layer_sizes=(2,),	
	activation="logistic", 
	solver="lbfgs", 
	# Maximum try for optimization.
	max_iter=10000, 
	# Seed for random generator used in the algorithm.
	# Specifying this is useful to reproduce the result.
	random_state=1
)

# model.fit() method requires 2D array for x.
# array.reshape(-1, 1) will convert 1D array to 2D like this:
# Original: [1, 2, ,3]
# After reshape: [[1], [2], [3]]
# First parameter of reshape represents the total size of the new array.
# -1 is a magic number meaning that the total size of the new array is the same as the original array's total size.
# Second parameter of reshape represents the size of each element in the new array.
x_2d = np.array(x).reshape(-1, 1)

model.fit(x_2d, y)

# It will print the following:
# Weights: [array([[-6.38637576, -4.35627248]]), array([[-15.70729999],[ 15.93923995]])]
# Biases: [array([21.89251891, 29.70956915]), array([-8.30789949])]
print(f"Weights: {model.coefs_}")
print(f"Biases: {model.intercepts_}")

# It will print this:
# Probabilities for x = 5: [[4.87900059e-04 9.99512100e-01]]
# First number is the probability that the data belongs to group 0.
# Second number is the probability that the data belongs to group 1.
# model.predict_proba takes 2D array as an input.
print(f"Prediction for x = 5: {model.predict_proba([[5]])}")

# Let the model classify the data.
# It will print this:
# The model predicts x = 5 belongs to : [1]
print(f"The model predicts x = 5 belongs to : {model.predict([[5]])}")

# The below is just for the show.

# Data that belongs to group 0.
x_0 = [0.5, 1, 1.5, 2, 2.5, 7.5, 8, 8.5, 9, 9.5]
y_0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
plt.scatter(x_0, y_0, color="red")

# Data that belongs to group 1.
x_1 = [4, 4.5, 5, 5.5, 6]
y_1 = [1, 1, 1, 1, 1]
plt.scatter(x_1, y_1, color="green")

# Generate evenly spaced 1000 numbers from 0 to 10 (all-inclusive).
model_x = np.linspace(0, 10, 1000)
model_probabilities = model.predict_proba(np.array(model_x).reshape(-1, 1))
# Make the array of probability that the data belongs to group 1.
model_y = [item[1] for item in model_probabilities]

plt.plot(model_x, model_y)
plt.show()

```

## StandardScaler

Here are the effects:
- The mean becomes zero (μ = 0).
- The standard deviation becomes one (σ = 1).

Here is the question:<br>
`z = (x - μ) / σ`, where z is the result of the scaling.

---

Pros:
- It makes the dataset easier to compare the relative importance of feature values.
  For example, let's say we have the following dataset:
  | Income | Age |
  | ------ | --- |
  | 60000  | 30  |
  | 75000  | 35  |
  | 40000  | 22  |
  | 90000  | 45  |
  | 55000  | 28  |
  
  We apply the `StandardScaler` like this:
  ```python
  from sklearn.preprocessing import StandardScaler

  data = [[60000, 30], [75000, 35], [40000, 22], [90000, 45], [55000, 28]]
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(data)
  ```
  
  The standardized data would look like this:
  ```
  [[-0.23328474 -0.25906388]
  [ 0.64153303  0.38859582]
  [-1.39970842 -1.2953194 ]
  [ 1.51635079  1.68391522]
  [-0.52489066 -0.51812776]]
  ```
  Since both income and age are on the same scale, it's easier to see their relative importance.
- It centers the data by removing the mean, which can be important for algorithms that assume the dataset is centered like a normal distribution.
- Since it operates independently on each feature, there is no leakage from the test set into the training set during scaling.
  The term `leakage` refers to a situation where the dataset for testing is somehow included during the training. The leakage may cause overestimation.

Cons:
 - Since the mean and standard deviation can be heavily influenced by outliers, it might not be suitable for a dataset with extreme outliers. We may want to use the `RobustScaler` or other scalers instead.
 - The original units and scales are lost, which might be important for some situations.
 - Since it assumes the dataset is in Gaussian (normal) distribution, it might not be suitable for non-normal distributions.
