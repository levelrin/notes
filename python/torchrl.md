## ProbabilisticActor

```python
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, InteractionType
from torch import nn
from torch.distributions import Categorical
from torchrl.modules import ProbabilisticActor


def main():
    # Let's say there are 2 states and 4 possible actions.
    linear = nn.Linear(2, 4)
    # The "states" in the `in_keys` is arbitrarily decided. Anything is okay.
    # However, we need to put specific key names for the `out_keys` for `ProbabilisticActor`.
    # Specifically, it depends on what distribution we choose via `distribution_class`.
    # To figure out the appropriate key names, we need to check the distribution's documentation.
    # It should say something like the distribution is "parameterized by ..." attributes.
    # The attribute name(s) is(are) the `output_keys` we should put.
    # For example, we chose the `Categorical` distribution because we are dealing with discrete actions.
    # The `Categorical` distribution is parameterized by either `probs` or `logits` attribute.
    # Since we are using a raw score rather than a probability distribution, we use the "logits" key.
    # By the way, the phrase "parameterized by" is the same as the phrase "characterized by".
    # FYI, people often use the normal distribution when they are dealing with continuous actions.
    # The normal distribution is parameterized by `loc` and `scale` attributes.
    # The `loc` stands for location, and it means an average (mean).
    # The `scale` means a standard deviation.
    tensor_dict_module = TensorDictModule(linear, in_keys=["states"], out_keys=["logits"])
    actor = ProbabilisticActor(
        module=tensor_dict_module,
        # The output from the `TensorDictModule` will be used as an input for `ProbabilisticActor`.
        in_keys=["logits"],
        distribution_class=Categorical,
        # The default is `InteractionType.DETERMINISTIC`.
        # However, the `Categorical` distribution does not have the `deterministic_sample` attribute.
        # That's why we use the mode, which means we pick the index with the highest value among the output nodes.
        default_interaction_type=InteractionType.MODE
    )
    tensor_dict = TensorDict({
        "states": torch.FloatTensor([[0, 1], [2, 3], [4, 5]])
    }, batch_size=3)
    # It will have the `action` key by default unless we specify the `out_keys` in the actor's constructor.
    # As the name suggests, the action is the decision made by the actor.
    # In this case, the actor just picks the action with the highest score.
    output_tensor_dict = actor(tensor_dict)
    print(f"output_tensor_dict: {output_tensor_dict}")
    logits = output_tensor_dict["logits"]
    print(f"logits: {logits}")
    action = output_tensor_dict["action"]
    print(f"action: {action}")

main()
```

Here is the console output:
```
output_tensor_dict: TensorDict(
    fields={
        action: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.int64, is_shared=False),
        logits: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
        states: Tensor(shape=torch.Size([3, 2]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([3]),
    device=None,
    is_shared=False)
logits: tensor([[ 0.7498, -0.1417,  0.3959, -0.0788],
        [ 0.5097, -1.3206,  1.7100,  0.0444],
        [ 0.2695, -2.4994,  3.0241,  0.1675]], grad_fn=<AddmmBackward0>)
action: tensor([0, 2, 2])
```
