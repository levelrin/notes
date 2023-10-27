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

## KNeighborsRegressor

TBD.

## LinearRegression

TBD.

## StandardScaler

Here are the effects:
- The mean becomes zero (μ = 0).
- The standard deviation becomes one (σ = 1).

Here is the question:<br>
`z = (x - μ) / σ`, where z is the result of the scaling.
