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
