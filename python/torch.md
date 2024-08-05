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
# I used this command to install the correct one:
# ```sh
# pip install -f https://download.pytorch.org/whl/torch_stable.html torch
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
