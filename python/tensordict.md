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
