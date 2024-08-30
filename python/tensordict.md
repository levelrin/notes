## TensorDict and TensorDictModule

```python
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn


def main():
    # Shape: (3, 1).
    initial_tensor = torch.FloatTensor([[0], [1], [2]])

    # All tensors in the TensorDict must have the same batch size.
    tensor_dict = TensorDict({
        "input": initial_tensor
    }, batch_size=3)

    # A simple linear model.
    linear = nn.Linear(1, 2)

    # TensorDictModule is a wrapper for nn.Module.
    # The input for the linear model will be mapped to TensorDict with the key "input".
    # Likewise, the output from the linear model will be mapped to TensorDict with the key "output".
    # Although the analyzer complains that the expected type for `out_keys` is list[NestedKey],
    # The list[NestedKey] is the same thing as the list of strings.
    tensor_dict_module = TensorDictModule(linear, in_keys=["input"], out_keys=["output"])

    # TensorDictModule(TensorDict(Tensor)) -> TensorDict.
    output_tensor_dict = tensor_dict_module(tensor_dict)
    print(f"output_tensor_dict: {output_tensor_dict}")

    # We can get the tensor from the TensorDict using the key.
    # By the way, `input_tensor` and `initial_tensor` are the same.
    input_tensor = output_tensor_dict["input"]
    print(f"input_tensor: {input_tensor}, initial_tensor: {initial_tensor}, equal: {torch.equal(input_tensor, initial_tensor)}")

    # This is the output from the linear model.
    output_tensor = output_tensor_dict["output"]
    print(f"output_tensor: {output_tensor}")

    # We can even set the tensor using the key.
    # Note that it won't change `initial_tensor` or `input_tensor`.
    output_tensor_dict["input"] = torch.FloatTensor([[3], [4], [5]])
    input_tensor_2 = output_tensor_dict["input"]
    print(f"input_tensor_2: {input_tensor_2}, initial_tensor: {initial_tensor}, equal: {torch.equal(input_tensor_2, initial_tensor)}")

main()
```

Here is the output:
```
output_tensor_dict: TensorDict(
    fields={
        input: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        output: Tensor(shape=torch.Size([3, 2]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([3]),
    device=None,
    is_shared=False)
input_tensor: tensor([[0.],
        [1.],
        [2.]]), initial_tensor: tensor([[0.],
        [1.],
        [2.]]), equal: True
output_tensor: tensor([[ 0.2114, -0.7739],
        [ 1.1811, -1.2224],
        [ 2.1508, -1.6708]], grad_fn=<AddmmBackward0>)
input_tensor_2: tensor([[3.],
        [4.],
        [5.]]), initial_tensor: tensor([[0.],
        [1.],
        [2.]]), equal: False
```

## TensorDictModule with multiple keys

```python
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn


class OurModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(1, 2)
        self.linear_2 = nn.Linear(1, 2)

    def forward(self, inputs_1, inputs_2):
        """
        Calculate the average of two linear outputs.
        :param inputs_1: A tensor with the shape: (batch_size, 1).
        :param inputs_2: A tensor with the shape: (batch_size, 1).
        :return: A tensor with the shape: (batch_size, 2).
        """
        outputs_1 = self.linear_1(inputs_1)
        outputs_2 = self.linear_2(inputs_2)
        return (outputs_1 + outputs_2) / 2

def main():
    inputs_tensor_1 = torch.FloatTensor([[0], [1], [2]])
    inputs_tensor_2 = torch.FloatTensor([[3], [4], [5]])
    model = OurModel()
    tensor_dict = TensorDict({
        "inputs_1": inputs_tensor_1,
        "inputs_2": inputs_tensor_2
    }, batch_size=3)
    # Since our model takes 2 inputs, we can have two in_keys.
    # TensorDictModule will map the keys and inputs in order.
    tensor_dict_module = TensorDictModule(model, in_keys=["inputs_1", "inputs_2"], out_keys=["output"])
    output_tensor_dict = tensor_dict_module(tensor_dict)
    print(f"output_tensor_dict: {output_tensor_dict}")

    # We can confirm the input mapping.
    inputs_1 = output_tensor_dict["inputs_1"]
    print(f"inputs_1: {inputs_1}")
    inputs_2 = output_tensor_dict["inputs_2"]
    print(f"inputs_2: {inputs_2}")

main()
```

Here is the output:
```
output_tensor_dict: TensorDict(
    fields={
        inputs_1: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        inputs_2: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        output: Tensor(shape=torch.Size([3, 2]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([3]),
    device=None,
    is_shared=False)
inputs_1: tensor([[0.],
        [1.],
        [2.]])
inputs_2: tensor([[3.],
        [4.],
        [5.]])
```
