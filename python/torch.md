## Tensor

```python
import torch

tensor1 = torch.LongTensor([0, 1, 2, 3])
tensor2 = torch.tensor([0, 1, 2, 3], dtype=torch.long)
tensor3 = torch.FloatTensor([0, 1, 2, 3])
tensor4 = torch.tensor([0, 1, 2, 3], dtype=torch.float32)
print(f"tensor1: {tensor1}")
print(f"tensor2: {tensor2}")
print(f"tensor3: {tensor3}")
print(f"tensor3 type: {type(tensor3)}")
print(f"tensor3 dtype: {tensor3.dtype}")
print(f"dtype comparison between tensor1 and tensor2: {tensor1.dtype == tensor2.dtype}")
print(f"dtype comparison between tensor1 and tensor3: {tensor1.dtype == tensor3.dtype}")
print(f"dtype comparison between tensor3 and tensor4: {type(tensor3) == type(tensor4)}")
print(f"type comparison between tensor1 and tensor3: {type(tensor1) == type(tensor3)}")
```

Output of the above code:
```
tensor1: tensor([0, 1, 2, 3])
tensor2: tensor([0, 1, 2, 3])
tensor3: tensor([0., 1., 2., 3.])
tensor3 type: <class 'torch.Tensor'>
tensor3 dtype: torch.float32
dtype comparison between tensor1 and tensor2: True
dtype comparison between tensor1 and tensor3: False
dtype comparison between tensor3 and tensor4: True
type comparison between tensor1 and tensor3: True
```

### Tensor.unsqueeze

```python
import torch

# We can say that the shape of this tensor is (4).
# In other words, it's a 1D tensor with 4 elements inside.
# The Tensor.unsqueeze(n) increases the dimension by 1.
# In this case, Tensor.unsqueeze(n) will make this a 2D tensor.
# The parameter `n` represents the index of the shape.
original = torch.LongTensor([0, 1, 2, 3])
print(f"original: {original}")

# It means adding one dimension at index=0 of the shape.
# In this case, the shape will be: (4) -> (1, 4).
# The shape (1, 4) means 1 element outside and 4 elements inside.
dim0 = original.unsqueeze(0)
print(f"dim=0: {dim0}")

# It means adding one dimension at index=1 of the shape.
# In this case, the shape will be: (4) -> (4, 1).
# The shape (4, 1) means 4 elements outside and 1 element inside.
dim1 = original.unsqueeze(1)
print(f"dim=1: {dim1}")

# A negative `n` means we count the index from the end (opposite direction).
# In this case, the shape will be: (4) -> (4, 1).
dim_1 = original.unsqueeze(-1)
print(f"dim=-1: {dim_1}")

# In this case, the shape will be: (4) -> (1, 4).
dim_2 = original.unsqueeze(-2)
print(f"dim=-2: {dim_2}")

# So, the range of dim (n) would be [-2, 1] in this case.
# Other values will result in an error (dimension out of range).
```

Output of the above code:
```
original: tensor([0, 1, 2, 3])
dim=0: tensor([[0, 1, 2, 3]])
dim=1: tensor([[0],
        [1],
        [2],
        [3]])
dim=-1: tensor([[0],
        [1],
        [2],
        [3]])
dim=-2: tensor([[0, 1, 2, 3]])
```

### Tensor.squeeze

```python
import torch

# Shape is (1, 2, 2, 1, 3).
original = torch.LongTensor([
    [
        [
            [
                [0, 1, 2]
            ],
            [
                [3, 4, 5]
            ]
        ],
        [
            [
                [6, 7, 8]
            ],
            [
                [9, 10, 11]
            ]
        ],
    ]
])
print(f"original: {original}")
print(f"original's shape: {original.shape}")

# The squeeze method is used to reduce the dimension of a tensor.
# Calling squeeze method without `dim` parameter removes all dimensions of shape=1.
# In this case, the shape will be: (1, 2, 2, 1, 3) -> (2, 2, 3).
simple_squeeze = original.squeeze()
print(f"simple_squeeze: {simple_squeeze}")
print(f"simple_squeeze's shape: {simple_squeeze.shape}")

# Specifying `dim` would remove the dimension at that specific index if its shape=1.
# In this case, the shape will be: (1, 2, 2, 1, 3) -> (1, 2, 2, 3).
dim3 = original.squeeze(dim=3)
print(f"dim=3: {dim3}")
print(f"dim=3 shape: {dim3.shape}")

# The shape of the dim (index) = 4 is 3.
# Since the shape at dim=4 is not 1, it will not reduce that dimension.
# In this case, the shape will be: (1, 2, 2, 1, 3) -> (1, 2, 2, 1, 3).
dim4 = original.squeeze(dim=4)
print(f"dim=4: {dim4}")
print(f"dim=4 shape: {dim4.shape}")
```

Output of the above code (formatted):
```
original: tensor([[[[[0, 1, 2]], [[3, 4, 5]]], [[[6, 7, 8]], [[9, 10, 11]]]]])
original's shape: torch.Size([1, 2, 2, 1, 3])
simple_squeeze: tensor([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
simple_squeeze's shape: torch.Size([2, 2, 3])
dim=3: tensor([[[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]])
dim=3 shape: torch.Size([1, 2, 2, 3])
dim=4: tensor([[[[[0, 1, 2]], [[3, 4, 5]]], [[[6, 7, 8]], [[9, 10, 11]]]]])
dim=4 shape: torch.Size([1, 2, 2, 1, 3])
```

### torch.cat

```python
import torch

# Shape: (2, 3)
tensor1 = torch.LongTensor([[0, 1, 2], [3, 4, 5]])
# Shape: (2, 3)
tensor2 = torch.LongTensor([[6, 7, 8], [9, 10, 11]])
# The `dim` corresponds to the index that we use for the addition.
# Shape: (2, ignored) + (2, ignored) = (4, 3)
tensor3 = torch.cat((tensor1, tensor2), dim=0)
# Shape: (ignored, 3) + (ignored, 3) = (2, 6)
tensor4 = torch.cat((tensor1, tensor2), dim=1)
print(f"tensor1's shape: {tensor1.shape}")
print(f"tensor2's shape: {tensor2.shape}")
print(f"tensor3's shape: {tensor3.shape}")
print(f"tensor3: {tensor3}")
print(f"tensor4's shape: {tensor4.shape}")
print(f"tensor4: {tensor4}")
```

Output of the above code:
```
tensor1's shape: torch.Size([2, 3])
tensor2's shape: torch.Size([2, 3])
tensor3's shape: torch.Size([4, 3])
tensor3: tensor([[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11]])
tensor4's shape: torch.Size([2, 6])
tensor4: tensor([[ 0,  1,  2,  6,  7,  8],
        [ 3,  4,  5,  9, 10, 11]])
```

### torch.topk

```python
import torch

tensor1 = torch.FloatTensor([[8, 0, 1, 2, 3, 4, 5, 6, 7]])
top_values1, top_indexes1 = torch.topk(tensor1, 2)
print(f"top_values1: {top_values1}, top_indexes1: {top_indexes1}")

tensor2 = torch.FloatTensor([[8, 8, 8, 8, 3, 4, 5, 6, 7]])
top_values2, top_indexes2 = torch.topk(tensor2, 2)
print(f"top_values2: {top_values2}, top_indexes2: {top_indexes2}")
```

Output of the above code:
```
top_values1: tensor([[8., 7.]]), top_indexes1: tensor([[0, 8]])
top_values2: tensor([[8., 8.]]), top_indexes2: tensor([[0, 1]])
```

## CrossEntropyLoss

```python
import numpy as np
import torch
from torch import nn

# This function is useful for classification problems.
loss_function = nn.CrossEntropyLoss()
# We must use raw scores (logits) instead of the probability distribution.
# In other words, we don't have to apply softmax.
# In this case, there are 3 classes.
# Note that it's 2D because the cross-entropy function expects batch input.
raw_prediction1 = torch.FloatTensor([[2, 0, -1]])
# The target input must be the index of the class.
# In this case, the third class was the correct answer.
# Note that it's 1D, unlike above input.
actual_class1 = torch.LongTensor([2])
loss1 = loss_function(raw_prediction1, actual_class1)
print(f"loss1: {loss1}")

# This example shows the loss gets lower if the prediction is correct.
raw_prediction2 = torch.FloatTensor([[2, 0, -1]])
actual_class2 = torch.LongTensor([0])
loss2 = loss_function(raw_prediction2, actual_class2)
print(f"loss2: {loss2}")

# This example shows a case of batch input.
raw_prediction3 = torch.FloatTensor([[2, 0, -1], [2, 0, -1]])
# Note that it's still 1D.
actual_class3 = torch.LongTensor([2, 0])
# Since there are two losses (2 batches), the average loss value will be returned.
loss3 = loss_function(raw_prediction3, actual_class3)
print(f"loss3: {loss3}, average of loss1 and loss2: {np.mean([loss1, loss2])}")
```

Output of the above code:
```
loss1: 3.1698460578918457
loss2: 0.1698460429906845
loss3: 1.6698460578918457, average of loss1 and loss2: 1.6698460578918457
```

## MultiLabelSoftMarginLoss

```python
import numpy as np
import torch
from torch import nn

# This function is useful for multi-label classification tasks.
loss_function = nn.MultiLabelSoftMarginLoss()

# We must use raw scores (logits) instead of the probability distribution.
# In other words, we don't have to apply softmax.
# In this case, there are 3 classes.
# Note that it's 2D because it expects batch input.
raw_prediction1 = torch.FloatTensor([[2, 0, -1]])
# The target input must have the same shape as the raw prediction.
# We use one-hot encoded tensor to denote the correct labels.
# In this case, the first and the third labels are correct.
actual_classes1 = torch.LongTensor([[1, 0, 1]])
loss1 = loss_function(raw_prediction1, actual_classes1)
print(f"loss1: {loss1}")

# This example shows the loss gets lower if the prediction is more correct.
raw_prediction2 = torch.FloatTensor([[2, -1, 1.5]])
actual_classes2 = torch.LongTensor([[1, 0, 1]])
loss2 = loss_function(raw_prediction2, actual_classes2)
print(f"loss2: {loss2}")

# This example shows a case of batch input.
raw_prediction3 = torch.FloatTensor([[2, 0, -1], [2, -1, 1.5]])
actual_classes3 = torch.LongTensor([[1, 0, 1], [1, 0, 1]])
# Since there are two losses (2 batches), the average loss value will be returned.
loss3 = loss_function(raw_prediction3, actual_classes3)
print(f"loss3: {loss3}, average of loss1 and loss2: {np.mean([loss1, loss2])}")
```

Output of the above code:
```
loss1: 0.7111123204231262
loss2: 0.2138676643371582
loss3: 0.4624899923801422, average of loss1 and loss2: 0.4624899923801422
```

## Neural Networks with Raw Parameters

```python
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

import matplotlib.pyplot as plt


class OurNet(nn.Module):
    """
    Creating a neural network means creating a new class that extends PyTorch's `nn.Module` class.
    In this sample, we will create a neural network with one input node, one hidden layer with two nodes, and one output node.
    Here is the overview:
        ```
                  ↗ Hidden Node 1 ↘
        Input Node                  Output Node
                  ↘ Hidden Node 2 ↗
        ```
    Here is the detail:
        ```
         ↗ a_0 = activate(i * w_00 + b_00) ↘
        i                                    o = activate(a_0 * w_01 + a_1 * w_11 + b_final)
         ↘ a_1 = activate(i * w_10 + b_10) ↗
        ```
        i means input.
        w means weight.
        b means bias.
        o means output.
    """

    def __init__(self):
        """
        Constructor.
        Here, we can initialize the weights and biases of the neural network.
        """
        super().__init__()
        # Parameters for the hidden node 1.
        # We can set initial values.
        # Note that values must be floating points, not integers.
        # `requires_grad` determines whether we optimize the parameter or not.
        # We set `requires_grad` to False if it's already optimized.
        # Otherwise, we set it to True so that it will be updated during training.
        self.w_00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b_00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)

        # For the hidden node 2.
        self.w_10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b_10 = nn.Parameter(torch.tensor(0.), requires_grad=False)

        # For the output node.
        self.w_01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        self.w_11 = nn.Parameter(torch.tensor(2.70), requires_grad=False)
        self.b_final = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, i):
        """
        It processes the input and generates the output from the neural network.
        By the way, this method is inherited by PyTorch.
        :param i: Input
        :return: Output from the neural network.
        """
        a_0 = func.relu(i * self.w_00 + self.b_00)
        a_1 = func.relu(i * self.w_10 + self.b_10)
        return func.relu(a_0 * self.w_01 + a_1 * self.w_11 + self.b_final)

    def fit(self, train_inputs, train_labels):
        """
        This is a custom method, not one from PyTorch.
        :param train_inputs: As is.
        :param train_labels: Expected values.
        :return: Void.
        """
        # We will use the stochastic gradient descent.
        # self.parameters() has all the weights and biases we defined.
        # lr means learning rate.
        optimizer = optim.SGD(self.parameters(), lr=0.1)
        # epoch refers to an iteration in which the optimization process goes through all the training data.
        # For example, 100 epochs mean we will use all the training data 100 times for optimization.
        for epoch in range(100):
            total_loss = 0
            for index in range(len(train_inputs)):
                input_i = train_inputs[index]
                label_i = train_labels[index]
                output_i = self.forward(input_i)
                loss = (output_i - label_i) ** 2
                # Calculate the derivative of the loss.
                # Note that it will be saved internally in the model.
                # That means the derivative will add up, as we call the `backward` method.
                # I know it's not intuitive because it seems the `loss` variable is freshly created.
                # But note that the `loss` is coming from `output_i,` which is from the model.
                # For that reason, the derivative is somehow stored in the model.
                loss.backward()
                total_loss += float(loss)
            if total_loss < 0.0001:
                break
            # Update weights and biases based on the derivatives we have stored so far.
            optimizer.step()
            # Since we updated the parameters, we need to clear the derivatives for the next epoch.
            optimizer.zero_grad()


def main():
    model = OurNet()

    # Training data:
    train_inputs = torch.tensor([0., 0.5, 1.])
    train_labels = torch.tensor([0., 1., 0.])

    # Train
    model.fit(train_inputs, train_labels)

    # Generate evenly spaced 11 numbers from 0 to 1 (all-inclusive).
    input_values = torch.linspace(start=0, end=1, steps=11)
    # The model will call the `forward(i)` method by default.
    output_values = model(input_values)

    # Show the training data on the graph.
    plt.scatter(train_inputs, train_labels, color="red")

    # Show the regression line on the graph.
    # Tensor has a list of data and gradients.
    # plt.plot(x, y) doesn't know what to do with the gradient.
    # The `detach()` will remove that gradient and make it look like a normal list (it's still a tensor type, though).
    # Error will occur if `detach()` is not called.
    plt.plot(input_values.detach(), output_values.detach())
    plt.show()


main()
```

## Linear

```python
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

import matplotlib.pyplot as plt


class OurNet(nn.Module):
    """
    In this sample, we will create a neural network with one input node, one hidden layer with two nodes, and one output node.
    """

    def __init__(self):
        super().__init__()
        # Define layers.
        # nn.Linear is a fully connected layer.
        # The first parameter is the number of inputs.
        # The second parameter is the number of outputs.
        self.hidden_layer = nn.Linear(1, 2)
        self.output_layer = nn.Linear(2, 1)

        # Specify the initial weights and biases.
        # Random weights and biases will be used if we don't specify.
        # Also, requires_grad=True will be used by default.
        # Since the hidden layer has one input and two outputs, the weights are represented as a 2x1 matrix.
        # It means there are 2 rows (number of outputs) and 1 column (number of inputs).
        # In Python, a matrix is often represented as a 2D list.
        # The number of rows corresponds to the number of elements in the list.
        # The number of columns corresponds to the number of elements in each inner list.
        # Here is an example:
        # matrix_3x5 = [
        #     [1, 2, 3, 4, 5],  # First row
        #     [6, 7, 8, 9, 10],  # Second row
        #     [11, 12, 13, 14, 15]  # Third row
        # ]
        self.hidden_layer.weight = nn.Parameter(torch.tensor([[1.7], [12.6]]), requires_grad=False)
        # Bias is always 1D list.
        # The number of bias values depends on the number of outputs.
        self.hidden_layer.bias = nn.Parameter(torch.tensor([-0.85, 0.]), requires_grad=False)
        self.output_layer.weight = nn.Parameter(torch.tensor([[-40.8, 2.70]]), requires_grad=False)
        self.output_layer.bias = nn.Parameter(torch.tensor([0.]), requires_grad=True)

    def forward(self, i):
        # `self.hidden_layer(i)` implicitly calls its `forward` method.
        # The mathematical operation can be represented as follows:
        # y = x * A^T + b
        # y is the output vector.
        # The shape would be a `batch size` x `number of output nodes` matrix.
        # For example, we have inputs like this: [[0.1], [0.2], [0.3]].
        # Since there are 3 inputs, the batch size is 3.
        # So, y would be a 3x2 matrix.
        #
        # x is the input vector.
        # It's often a 2D tensor, in which the first dimension represents the batch size (number of data),
        # and the second dimension represents the number of inputs.
        # Since the hidden layer has 1 input node, the input tensor may look like this: [[0.1], [0.2], [0.3]], in which
        # the batch size is 3.
        #
        # A^T is the transpose of the weight matrix.
        # Since the weight matrix of the hidden layer is [[1.7], [12.6]], the transpose is [[1.7, 12.6]].
        #
        # b is the bias vector.
        # The bias vector of the hidden layer is [-0.85, 0.], which is a 1x2 matrix.
        # By the way, it seems there is a problem.
        # x is a 3x1 matrix (assuming the batch size is 3).
        # A^T is a 1x2 matrix.
        # So, x * A^T would be a 3x2 matrix.
        # Since x * A^T + b would be 3x2 matrix + 1x2 matrix, it seems we cannot add them (dimension mismatch).
        # Fortunately, PyTorch does something called broadcasting.
        # It's a technique to use tensors with different shapes by expanding the smaller ones to the necessary dimensions.
        # In this case, the bias will be expanded to a 3x2 matrix by duplicating the rows like this:
        # b = [[-0.85, 0.], [-0.85, 0.], [-0.85, 0.]]
        #
        # Note that no activation function is used in that operation.
        # In other words, it's effectively the same as using the identical (linear) function.
        # The relu function will simply be applied to each number in the tensor and keep its shape.
        hidden_output = func.relu(self.hidden_layer(i))
        return func.relu(self.output_layer(hidden_output))

    def fit(self, train_inputs, train_labels):
        loss_function = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=0.1)
        for epoch in range(100):
            outputs = self.forward(train_inputs)
            # It's implicitly calling MSELoss.forward(outputs, train_labels) method.
            loss = loss_function(outputs, train_labels)
            if loss < 0.0001:
                break
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def main():
    model = OurNet()

    # Training data.
    # Obviously, we should check the shapes of the inputs and outputs, one from the input layer and another from the output layer.
    # In this case, we are using Linear layers, in which the following mathematical operation is used:
    # y = x * A^T + b
    # y determines the shape of outputs.
    # x determines the shape of inputs.
    train_inputs = torch.tensor([[0.], [0.5], [1.]])
    train_labels = torch.tensor([[0.], [1.], [0.]])

    # Train
    model.fit(train_inputs, train_labels)

    # Draw the regression line.
    # tensor.reshape(-1, 1) will convert 1D tensor to 2D like this:
    # Original: [1, 2, 3]
    # After reshape: [[1], [2], [3]]
    # The first parameter of reshape represents the total size of the new array.
    # -1 is a magic number, meaning that the total size of the new array is the same as the original array's total size.
    # The second parameter of reshape represents the size of each element in the new array.
    input_values = torch.linspace(start=0, end=1, steps=11).reshape(-1, 1)
    output_values = model(input_values)
    # tensor.squeeze() removes all the dimensions if the size of an element is 1.
    # For example, [[1], [2], [3]] will be [1, 2, 3].
    # We are squeezing the tensor to plot them.
    plt.plot(input_values.squeeze().detach(), output_values.squeeze().detach())

    # Show the training data.
    plt.scatter(train_inputs, train_labels, color="red")
    plt.show()


main()
```

## Argmax

```python
import torch
from torch import nn

model = nn.Linear(3, 6)
input_tensor = torch.FloatTensor([[3, 6, 9]])
output = model(input_tensor)
print(f"output: {output}")
# The argmax will return the index of the maximum value.
# The input will be flattened. That's why it makes sense that we put 2D input.
max_index = torch.argmax(output)
print(f"max_index: {max_index}")
print(f"max_index type: {type(max_index)}")
print(f"max_index.item(): {max_index.item()}")
print(f"max_index.item() type: {type(max_index.item())}")
```

This was the output of the above code:
```
output: tensor([[ 0.8717,  1.8732, -0.9439, -3.9290,  0.4696, -1.9440]],
       grad_fn=<AddmmBackward0>)
max_index: 1
max_index type: <class 'torch.Tensor'>
max_index.item(): 1
max_index.item() type: <class 'int'>
```

## Visualize Optimization

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class OurNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Layers
        self.hidden_layer = nn.Linear(1, 20)
        self.output_layer = nn.Linear(20, 1)

        # Configuration for Optimization
        self.max_epoch = 500
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        # We will stop the optimization if the loss is less than this.
        self.tolerable_loss = 0.0001

        # Configurations for Plotting
        self.last_epoch = 0
        # `regression_x_start`, `regression_x_stop`, and `regression_x_num` are parameters for `np.linspace`.
        # It's for creating x values for the regression graph.
        self.regression_x_start = 0
        self.regression_x_stop = 1
        self.regression_x_num = 11
        self.fig, self.axs = plt.subplots(2, 1)
        plt.subplots_adjust(hspace=0.5)
        self.regression_plt = self.axs[0]
        self.regression_plt.set_title("Regression")
        # Store regression lines during the optimization.
        # The purpose is to see how regression line changes over epoch.
        self.regression_lines = []
        # It's for drawing each regression line in the graph.
        # We will use this on each animation update.
        self.each_regression_line = self.regression_plt.plot([], [])[0]
        # The purpose is to plot the loss curve.
        self.losses = []
        self.loss_plt = self.axs[1]
        self.loss_plt.set_title("Loss")
        self.loss_plt.set_xlabel("epochs")
        self.loss_plt.set_ylabel("loss")
        # It's for drawing the loss curve.
        # We will use this on each animation update.
        self.loss_line = self.loss_plt.plot([], [])[0]

    def forward(self, i):
        hidden_output = func.relu(self.hidden_layer(i))
        return func.relu(self.output_layer(hidden_output))

    def fit(self, train_inputs, train_labels):
        self.regression_plt.scatter(train_inputs, train_labels, color="red", label="Actual")
        for epoch in range(self.max_epoch):
            self.last_epoch = epoch

            # Create a regression line and store it.
            regression_inputs = torch.linspace(
                start=self.regression_x_start,
                end=self.regression_x_stop,
                steps=self.regression_x_num
            ).reshape(-1, 1)
            regression_outputs = self.forward(regression_inputs)
            self.regression_lines.append(regression_outputs.squeeze().detach())

            outputs = self.forward(train_inputs)
            loss = self.loss_function(outputs, train_labels)
            self.losses.append(loss.detach())
            if loss < self.tolerable_loss:
                break
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def create_animation(self):
        self.regression_plt.legend()
        self.loss_plt.set_xlim(0, self.last_epoch + 1)
        self.loss_plt.set_ylim(0, max(self.losses) + 0.05)

        def update(frame):
            self.loss_line.set_data(list(range(frame)), self.losses[:frame])
            np.linspace(1, 2, 3)
            regression_inputs = np.linspace(self.regression_x_start, self.regression_x_stop, self.regression_x_num)
            self.each_regression_line.set_data(regression_inputs, self.regression_lines[frame])
            return [self.loss_line, self.each_regression_line]

        return FuncAnimation(
            fig=self.fig,
            func=update,
            frames=len(self.losses),
            interval=50,
            repeat=False
        )


def main():
    train_inputs = torch.tensor([[0.], [0.5], [1.]])
    train_labels = torch.tensor([[0.], [1.], [0.]])
    model = OurNet()
    model.fit(train_inputs, train_labels)
    animation = model.create_animation()
    plt.show()


main()
```

## Check if GPU is used

```python
import torch

if torch.cuda.is_available():
    print("CUDA is available. PyTorch is using GPU.")
else:
    print("CUDA is not available. PyTorch is using CPU.")

# If this prints an empty list, you probably installed the cpu-only PyTorch.
# In that case, you need to uninstall the current torch and install the GPU-enabled one.
# I found the installation command for it from this URL: https://pytorch.org/get-started/locally
# Like this:
# ```sh
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# ```
# By the way, I recommend using a virtual environment so you don't have to mess with the system packages.
print(torch.cuda.get_arch_list())
```

## Seed Configuration

```python
import torch


# Set the seed for CPU.
# We can use any integer.
torch.manual_seed(0)

# Set the seed for all GPUs.
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
```

## Use GPU

```python
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt


seed = 1
device = torch.device("cpu")
torch.manual_seed(seed)
if torch.cuda.is_available():
    device = torch.device("cuda")
    # I noticed that this does not have any effect.
    # It seemed the model used CPU seed for some reason.
    # However, I was able to conclude that GPU was used because the computation time was quite different.
    torch.cuda.manual_seed_all(seed)


class OurNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
            nn.ReLU()
        )

    def forward(self, input_tensor):
        """
        As is.
        :param input_tensor: Shape: (batch_size, input_node_amount).
        :return: A tensor with the same shape as the input.
        """
        return self.fc(input_tensor)

    def fit(self, train_inputs, train_labels):
        """
        Train the model.
        :param train_inputs: A tensor with the shape: (batch_size, input_node_amount).
        :param train_labels: A tensor with the same shape as the `train_inputs`.
        :return: None.
        """
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        for epoch in range(500):
            optimizer.zero_grad()
            outputs = self.forward(train_inputs)
            loss = loss_function(outputs, train_labels)
            if loss < 0.0001:
                break
            loss.backward()
            optimizer.step()


def main():
    # Move the model to GPU.
    model = OurNet().to(device)

    # Training Data.
    # Make sure we move them to GPU.
    train_inputs = torch.tensor([[0.], [0.5], [1.]]).to(device)
    train_labels = torch.tensor([[0.], [1.], [0.]]).to(device)

    # Train
    model.fit(train_inputs, train_labels)

    # Draw the regression line.
    # We should move `input_values` to GPU because the model will use it.
    input_values = torch.linspace(start=0, end=1, steps=11).reshape(-1, 1).to(device)
    # We should move `output_values` to CPU because they are used for plotting.
    output_values = model(input_values).cpu()
    # We should move the `input_values` to CPU for plotting.
    plt.plot(input_values.squeeze().detach().cpu(), output_values.squeeze().detach())

    # Draw the training data to see the performance.
    # Again, we need to move all tensors to CPU for plotting.
    plt.scatter(train_inputs.cpu(), train_labels.cpu(), color="red")
    plt.show()

main()
```

## Save and Load Parameters

We can save the model's parameters like this:
```python
# This is the model that we made.
model = OurNet()
train_inputs = torch.tensor([[0.], [0.5], [1.]])
train_labels = torch.tensor([[0.], [1.], [0.]])
# Train the model.
model.fit(train_inputs, train_labels)

# It will serialize the model's parameters and save them to a file named `ournet_params.pth`.
# `.pth` is a file extension that the PyTorch community often uses for this situation.
# Apparently, `.pth` stands for `PyTorcH`.
torch.save(model.state_dict(), "ournet_params.pth")
```

And we can load the model's parameters like this:
```python
model.load_state_dict(torch.load("ournet_params.pth"))
```

## Embedding

```python
import torch
from torch import nn

# A map of tokens and their indexes.
# We can use this to convert a word into a vector.
token_to_index = {
    "<EOS>": 0,
    "roses": 1,
    "apples": 2,
    "red": 3,
    "limes": 4,
    "cucumbers": 5,
    "green": 6,
    "are": 7
}

# We can use this if we want to get the token using the index.
index_to_token = {index: token for token, index in token_to_index.items()}
print(f"index_to_token: {index_to_token}")

# `num_embeddings` is the number of tokens.
# `embedding_dim` is the number of dimensions of each token.
# `Embedding` object is just a lookup table
# in which each row represents each token,
# and each column represents each number in the token's vector.
embeds = nn.Embedding(num_embeddings=8, embedding_dim=2)

# We can check all the vectors like this:
print(embeds.weight)

# This is how we can get a vector of the first token.
first_token_vector = embeds(torch.LongTensor([0]))
print(f"The vector for the first token: {first_token_vector}")

# Create a lookup tensor for roses.
# We can use it as a key (index) for `embeds`.
lookup_roses = torch.tensor([token_to_index["roses"]], dtype=torch.long)
roses_vector = embeds(lookup_roses)
print(f"The vector for the roses: {roses_vector}")

# We can get a tensor with three vectors like this:
first_three_vectors = embeds(torch.LongTensor([0, 1, 2]))
print(f"The first three vectors: {first_three_vectors}")
```

### CBOW and Skip-Gram

```python
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim

torch.manual_seed(3)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

class OurEmbedding(nn.Module):

    def __init__(self, token_to_index):
        super().__init__()
        self.token_to_index = token_to_index
        self.index_to_token = {index: token for token, index in token_to_index.items()}
        self.vocab_size = len(token_to_index)
        token_vec_dim = 2
        self.embeds = nn.Embedding(self.vocab_size, token_vec_dim)
        # Fully connected layer (fc) for CBOW.
        self.fc_cbow = nn.Sequential(
            # FC: input layer -> hidden layer.
            nn.Linear(token_vec_dim * 2, token_vec_dim),
            nn.ReLU(),
            # FC: hidden layer -> output layer.
            nn.Linear(token_vec_dim, self.vocab_size),
            nn.ReLU()
        )
        # Fully connected layer for skip-gram.
        self.fc_skip_gram = nn.Sequential(
            nn.Linear(token_vec_dim, token_vec_dim),
            nn.ReLU(),
            nn.Linear(token_vec_dim, self.vocab_size),
            nn.ReLU()
        )

    def fit_with_cbow(self, sequences):
        """
        Use the continuous bag of words (CBOW) to train the embedding.
        It's about predicting the word in the middle.
        The window size is fixed to 2 for now.
        Ex: Given "roses are red", input: ["roses", "red"], output: "are".
        :param sequences: A simple list of sequences.
                          EX: [["roses", "are", "red"], ["limes", "are", "green"]]
        """
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(list(self.embeds.parameters()) + list(self.fc_cbow.parameters()), lr=0.01)
        for epoch in range(100):
            optimizer.zero_grad()
            total_loss = 0
            for sequence in sequences:
                seq_length = len(sequence)
                # Sequence too short.
                if seq_length < 3:
                    break
                for token_index in range(seq_length - 2):
                    token_left = sequence[token_index]
                    token_middle = sequence[token_index + 1]
                    token_right = sequence[token_index + 2]
                    token_left_vec = self.token_to_vec(token_left)
                    token_right_vec = self.token_to_vec(token_right)
                    # Input size: (1, token_vec_dim * 2)
                    fc_input = torch.cat((token_left_vec, token_right_vec), dim=1)
                    fc_output = self.fc_cbow(fc_input)
                    correct_token_index = self.token_to_index[token_middle]
                    loss = loss_function(fc_output, torch.LongTensor([correct_token_index]))
                    loss.backward()
                    total_loss += float(loss)

                    # Just for checking how's the prediction is going.
                    predicted_token_index = torch.argmax(fc_output).item()
                    predicted_token = self.index_to_token[predicted_token_index]
                    print(f"Prediction: {token_left} ({predicted_token}) {token_right}, Actual: {token_left} ({token_middle}) {token_right}")
            print(f"epoch: {epoch}, total_loss: {total_loss}\n")
            if total_loss < 0.0001:
                break
            optimizer.step()

    def fit_with_skip_gram(self, sequences):
        """
        Use the skip-gram to train the embedding.
        It's about predicting the surrounding words.
        The window size is fixed to 2 for now.
        Ex: Given "roses are red", input: "are", output: ["roses", "red"].
        :param sequences: Same as `fit_with_cbow`.
        """
        loss_function = nn.MultiLabelSoftMarginLoss()
        optimizer = optim.Adam(list(self.embeds.parameters()) + list(self.fc_skip_gram.parameters()), lr=0.01)
        for epoch in range(100):
            optimizer.zero_grad()
            total_loss = 0
            for sequence in sequences:
                seq_length = len(sequence)
                # Sequence too short.
                if seq_length < 3:
                    break
                for token_index in range(seq_length - 2):
                    token_left = sequence[token_index]
                    token_middle = sequence[token_index + 1]
                    token_right = sequence[token_index + 2]
                    token_middle_vec = self.token_to_vec(token_middle)
                    fc_output = self.fc_skip_gram(token_middle_vec)
                    token_left_index = self.token_to_index[token_left]
                    token_right_index = self.token_to_index[token_right]
                    # Initialize all-zero list and replace the value to 1 for correct indexes.
                    raw_correct_labels = [0] * self.vocab_size
                    raw_correct_labels[token_left_index] = 1
                    raw_correct_labels[token_right_index] = 1
                    correct_labels = torch.LongTensor([raw_correct_labels])
                    loss = loss_function(fc_output, correct_labels)
                    loss.backward()
                    total_loss += float(loss)

                    # Just for checking how's the prediction is going.
                    _, predicted_token_indexes = torch.topk(fc_output, 2)
                    predicted_token_index_left = predicted_token_indexes[0][0].item()
                    predicted_token_index_right = predicted_token_indexes[0][1].item()
                    predicted_token_left = self.index_to_token[predicted_token_index_left]
                    predicted_token_right = self.index_to_token[predicted_token_index_right]
                    print(f"Prediction: ({predicted_token_left}) {token_middle} ({predicted_token_right}), Actual: ({token_left}) {token_middle} ({token_right})")
            print(f"epoch: {epoch}, total_loss: {total_loss}\n")
            if total_loss < 0.0001:
                break
            optimizer.step()

    def token_to_vec(self, token):
        """
        Convert the token to vector.
        :param token: Ex: "roses"
        :return: Vector. Ex: tensor([[ 0.0299, -0.0498]], grad_fn=<EmbeddingBackward0>)
        """
        token_index = self.token_to_index[token]
        return self.embeds(torch.LongTensor([token_index]))

    def vec_to_token(self, vec):
        """
        Convert the vector to token.
        :param vec: Ex: tensor([[ 0.0299, -0.0498]], grad_fn=<EmbeddingBackward0>)
        :return: Token. Ex: "roses"
        """
        all_vecs = self.embeds.weight
        for each_vec in all_vecs:
            if torch.all(torch.eq(each_vec, vec)):
                return each_vec
        raise ValueError(f"Could not find the vector in the embedding. vector: {vec}, embedding: {all_vecs}")

def scatter(token_to_index, our_embedding, ax):
    """
    Visualize the token vectors.
    Assuming vector dimension is 2.
    :param token_to_index: As is.
    :param our_embedding: As is.
    :param ax: ax associated with the figure.
    """
    for token, index in token_to_index.items():
        # Squeeze to change the shape: (1, 2) -> (2)
        vec = our_embedding.token_to_vec(token).squeeze()
        x = vec[0].item()
        y = vec[1].item()
        ax.scatter(x, y)
        ax.text(x, y, token, horizontalalignment="left", verticalalignment="bottom", weight="semibold")

def main():
    # For visualizing the effect of the training.
    figs, axes = plt.subplots(1, 2)
    plt.subplots_adjust(wspace=0.5)
    ax_before = axes[0]
    ax_before.set_title("Before Training")
    ax_after = axes[1]
    ax_after.set_title("After Training")

    token_to_index = {
        # sos stands for start of sequence.
        "<sos>": 0,
        # eos stands for end of sequence.
        "<eos>": 1,
        "roses": 2,
        "apples": 3,
        "limes": 4,
        "cucumbers": 5,
        "are": 6,
        "red": 7,
        "green": 8
    }
    our_embedding = OurEmbedding(token_to_index)
    # Visualize the embedding before training.
    scatter(token_to_index, our_embedding, ax_before)
    training_sequences = [
        ["<sos>", "roses", "are", "red", "<eos>"],
        ["<sos>", "apples", "are", "red", "<eos>"],
        ["<sos>", "limes", "are", "green", "<eos>"],
        ["<sos>", "cucumbers", "are", "green", "<eos>"]
    ]
    #our_embedding.fit_with_cbow(training_sequences)
    our_embedding.fit_with_skip_gram(training_sequences)

    # Visualize the embedding after training.
    scatter(token_to_index, our_embedding, ax_after)
    plt.show()

main()
```

## Transformer

The official document about `nn.Transformer.forward(src, tgt)` says:
> src: (S, E) for unbatched input, (S, N, E) if batch_first=False or (N, S, E) if batch_first=True.
> 
> tgt: (T, E) for unbatched input, (T, N, E) if batch_first=False or (N, T, E) if batch_first=True.
>
> output: (T, E) for unbatched input, (T, N, E) if batch_first=False or (N, T, E) if batch_first=True.

`S` means the sequence length of the source input. In NLP, it's the number of words (tokens) in each sentence.

`T` is the same as `S`, except it's for the target input.

`E` means the embedding dimension. In NLP, it's the vector dimension of each word (token).

`N` means the batch size. In NLP, it's the number of sentences.

For example, `batch_first=True`, `S=3`, `E=2`, `N=5`, and `T=4`.

In that case, the `src` and `tgt` would look like the following:
```python
src = torch.tensor([
    [[1, 2], [3, 4], [5, 6]],  # Sequence 1
    [[7, 8], [9, 10], [11, 12]],  # Sequence 2
    [[13, 14], [15, 16], [17, 18]],  # Sequence 3
    [[19, 20], [21, 22], [23, 24]],  # Sequence 4
    [[25, 26], [27, 28], [29, 30]]   # Sequence 5
], dtype=torch.float32)

tgt = torch.tensor([
    [[31, 32], [33, 34], [35, 36], [37, 38]],  # Sequence 1
    [[39, 40], [41, 42], [43, 44], [45, 46]],  # Sequence 2
    [[47, 48], [49, 50], [51, 52], [53, 54]],  # Sequence 3
    [[55, 56], [57, 58], [59, 60], [61, 62]],  # Sequence 4
    [[63, 64], [65, 66], [67, 68], [69, 70]]   # Sequence 5
], dtype=torch.float32)
```

Since the `output` has the same shape as the `tgt`, the output of the above input would look like the following:
```
tensor([
    [[1.0000, -1.0000], [1.0000, -1.0000], [1.0000, -1.0000], [1.0000, -1.0000]],
    [[1.0000, -1.0000], [1.0000, -1.0000], [1.0000, -1.0000], [1.0000, -1.0000]],
    [[1.0000, -1.0000], [1.0000, -1.0000], [1.0000, -1.0000], [1.0000, -1.0000]],
    [[1.0000, -1.0000], [1.0000, -1.0000], [-1.0000, 1.0000], [-1.0000, 1.0000]],
    [[1.0000, -1.0000], [1.0000, -1.0000], [1.0000, -1.0000], [1.0000, -1.0000]]
], grad_fn=<NativeLayerNormBackward0>)
```

Example:
```python
import torch
import torch.nn as nn
import torch.optim as optim


seed = 8
device = torch.device("cpu")
torch.manual_seed(seed)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(seed)


class OurModel(nn.Module):

    def __init__(self, token_to_index):
        super().__init__()
        self.token_to_index = token_to_index
        self.index_to_token = {index: token for token, index in token_to_index.items()}
        self.vocab_size = len(token_to_index)
        # FYI, 512 is the default for the transformer.
        token_vec_dim = 2
        self.embeds = nn.Embedding(self.vocab_size, token_vec_dim)
        self.transformer = nn.Transformer(
            d_model=token_vec_dim,
            # `d_model` must be divisible by `nhead`.
            # FYI, default `nhead` is 8.
            # Also, you will get warnings if you set `nhead` to an odd number.
            nhead=8 if token_vec_dim % 8 == 0 else 2,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(token_vec_dim, token_vec_dim),
            nn.ReLU(),
            nn.Linear(token_vec_dim, self.vocab_size),
            nn.ReLU()
        )

    def forward(self, sequence, max_iteration=10):
        """
        Generate a sequence of tokens in an autoregressive way.
        The generation ends when <eos> token is generated or reaches the maximum iteration.
        Here is how it works:
        First, we put convert tokens into vectors.
        Second, we need two parameters to use `nn.Transformer(src, tgt)`.
        The `src` represents the input vectors.
        Ex: A tensor (of vectors) for ["roses", "are"].
        The `tgt` represents the generated vectors so far.
        Initially, we use the tensor for ["<sos>"].
        The output of `nn.Transformer(src, tgt)` is a tensor with the same shape as the `tgt`.
        For example, if `tgt` was a tensor for ["<sos>"], the output will be a tensor of vectors like [[1, 2]].
        We grab the [1, 2], put it into the fully connected layer to predict the next token.
        Once we got the next token, we append it to the `tgt`.
        For example, the `tgt` for the next iteration could be a tensor for ["<sos>", "red"].
        The transformer's output may look like a tensor of vectors like [[1, 2], [3, 4]].
        Note that we only grab the last element [3, 4] and ignore the rest.
        We put the last element into the fully connected layer to predict the next token.
        If the predicted token is "<eos>", we stop the generation.
        :param sequence: List of tokens. Ex: ["roses", "are"].
        :param max_iteration: As is.
        :return: List of tokens. Ex: ["<sos>", "red", "<eos>"].
        """
        token_indexes_raw = [self.token_to_index[token] for token in sequence]
        # Shape: (token_amount).
        token_indexes_tensor = torch.LongTensor([index for index in token_indexes_raw]).to(device)
        # Shape: (token_amount, token_vec_dim).
        token_vectors = self.embeds(token_indexes_tensor)
        # Shape: (1, token_vec_dim).
        sos_token_vec = self.embeds(torch.LongTensor([0]).to(device))
        # Shape: (1, token_vec_dim).
        eos_token_vec = self.embeds(torch.LongTensor([1]).to(device))
        # Shape: (generated_token_amount_so_far, token_vec_dim).
        current_target = sos_token_vec
        generated_token_indexes = [0]
        iteration = 1
        predicted_token_index = -1
        while predicted_token_index != 1 and iteration < max_iteration:
            # The shape is the same as the target.
            transformer_output = self.transformer(token_vectors, current_target)
            # Shape: (token_vec_dim).
            last_element = transformer_output[-1]
            # Shape: (vocab_size)
            raw_scores = self.fc(last_element)
            predicted_token_index = torch.argmax(raw_scores).item()
            generated_token_indexes.append(predicted_token_index)
            # Shape: (1, token_vec_dim).
            last_generated_token_vec = self.embeds(torch.LongTensor([predicted_token_index]).to(device))
            current_target = torch.cat((current_target, last_generated_token_vec), dim=0).to(device)
            iteration += 1
        return [self.index_to_token[index] for index in generated_token_indexes]

    def fit(self, sources, targets):
        """
        Train the model.
        Here is how it works:
        First, we convert tokens into vectors.
        Second, we get the output from `nn.Transformer(sources, targets)`.
        The transformer's output will be a batch of generated tokens.
        Note that the output's shape is the same as the shape of targets.
        Then, we put the transformer's output as an input for the fully connected layer.
        The fully connected layer will give a batch of raw scores.
        With each raw scores, we can determine the predicted next token.
        For the loss, we can compare the predicted next token and the actual next token from the target of the transformer.
        For example, the targets were: [["<sos>", "red"], ["<sos>", "green"]].
        The transformer's output was: [[value_1, value_2], [value_3, value_4]].
        The fully connected layer's output was: [[raw_score_1, raw_score_2], [raw_score_3, raw_score_4]].
        The raw_score_1 should've predicted the token "red".
        The raw_score_2 should've predicted the token "<eos>".
        As you can see, the last score corresponds to the "<eos>" token.
        That's why we excluded the "<eos>" token in targets.
        :param sources: A batch of sources in a plain list. Ex: [["roses", "are"], ["limes", "are"]].
        :param targets: A batch of targets in a plain list. Ex: [["red"], ["green"]].
                        Actually, it's supposed to look like this: [["<sos>", "red"], ["<sos>", "green"]].
                        However, this method will put "<sos>" for you.
                        Note that we exclude "<eos>" token on purpose.
        :return: None.
        """
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        for epoch in range(100):
            batch_source_indexes = [[self.token_to_index[token] for token in source] for source in sources]
            # Shape: (batch_size, token_amount_in_sequence, token_vec_dim).
            batch_source_vectors = self.embeds(torch.LongTensor(batch_source_indexes).to(device))
            # Note that we add the index of the "<sos>" token at the beginning for each target token.
            batch_target_indexes = [[0] + [self.token_to_index[token] for token in target] for target in targets]
            # Shape: (batch_size, token_amount_in_sequence, token_vec_dim).
            batch_target_vectors = self.embeds(torch.LongTensor(batch_target_indexes).to(device))
            # The shape is the same as the target.
            transformer_output = self.transformer(batch_source_vectors, batch_target_vectors)
            # Shape: (batch_size, token_amount_in_sequence, vocab_size).
            batch_raw_scores = self.fc(transformer_output)
            # Note that we add the index of the "<eos>" token at the end for each target token.
            batch_correct_token_indexes = [[self.token_to_index[token] for token in target] + [1] for target in targets]
            # Reshape to (batch_size * token_amount_in_sequence, vocab_size).
            # The reason is that the loss function requires 2D tensor as an input.
            # It's okay to reshape it because it's effectively the same thing as iterating the batch.
            # view method is the same as reshape method, except it's mutable.
            # The first -1 is a magic number for automatically calculate the appropriate size of the first dimension based on other dimension.
            # The second -1 means the size of the last dimension.
            # Let's say the original shape is (4, 2, 9).
            # The second -1 determined the size of the last dimension, which is 9.
            # That makes the remaining dimension to be 4 * 2 = 8 because view and reshape methods keep the number of data.
            # As a result, the new shape will be (8, 9).
            flatten_raw_scores = batch_raw_scores.view(-1, batch_raw_scores.size(-1))
            # Shape: (batch_size * token_amount_in_sequence).
            flatten_correct_token_indexes_tensor = torch.LongTensor(batch_correct_token_indexes).to(device).view(-1)
            loss = loss_function(flatten_raw_scores, flatten_correct_token_indexes_tensor)
            print(f"epoch: {epoch}, loss: {loss}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def main():
    token_to_index = {
        # sos stands for start of sequence.
        "<sos>": 0,
        # eos stands for end of sequence.
        "<eos>": 1,
        "roses": 2,
        "apples": 3,
        "limes": 4,
        "cucumbers": 5,
        "are": 6,
        "red": 7,
        "green": 8
    }

    model = OurModel(token_to_index).to(device)

    train_sources = [
        ["roses", "are"],
        ["apples", "are"],
        ["limes", "are"],
        ["cucumbers", "are"]
    ]
    train_targets = [
        ["red"],
        ["red"],
        ["green"],
        ["green"]
    ]
    model.fit(train_sources, train_targets)

    input_tokens = ["roses", "are"]
    generated_tokens = model(input_tokens)
    print(f"input_tokens: {input_tokens}, generated_tokens: {generated_tokens}")

main()
```
