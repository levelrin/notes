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
