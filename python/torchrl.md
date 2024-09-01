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

## ValueOperator

```python
import torch
from tensordict import TensorDict
from torch import nn
from torchrl.modules import ValueOperator


def main():
    # Let's say there are 2 states, and it will give the score of an action.
    value_net = nn.Linear(2, 1)
    value_operator = ValueOperator(
        module=value_net,
        # It's the parameters for model.forward() in order.
        # In this case, it's the same as the value_net.forward(tensor_dict["observation"]).
        in_keys=["observation"],
        # It's the outputs from the model.forward() in order.
        out_keys=["state_value"]
    )
    tensor_dict = TensorDict({
        "observation": torch.FloatTensor([[0, 1], [2, 3]])
    }, batch_size=2)
    output_tensor_dict = value_operator(tensor_dict)
    print(f"output_tensor_dict: {output_tensor_dict}")
    state_value = output_tensor_dict["state_value"]
    print(f"state_value: {state_value}")

main()
```

Here is the console output:
```
output_tensor_dict: TensorDict(
    fields={
        observation: Tensor(shape=torch.Size([2, 2]), device=cpu, dtype=torch.float32, is_shared=False),
        state_value: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([2]),
    device=None,
    is_shared=False)
state_value: tensor([[0.2035],
        [0.8668]], grad_fn=<AddmmBackward0>)
```

## GAE

```python
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.objectives.value import GAE


def main():
    # Let's say there are 2 states, and it will give the score of an action.
    value_network = nn.Linear(2, 1)
    tensor_dict_module = TensorDictModule(value_network, in_keys=["states"], out_keys=["value"])
    # It stands for generalized advantage estimation.
    gae = GAE(
        # Discount factor for future rewards.
        # Range: (0, 1] because gamma=0 can cause the "index out of range" error.
        # gamma -> 0 means we consider immediate rewards only.
        gamma=0.98,
        # It controls the trade-off between bias and variance in the estimation of advantages.
        # Range: (0, 1].
        # lmbda -> 0 means we are chasing for immediate advantages only (high bias and low variance).
        # lmbda -> 1 as we prefer long-term advantages.
        lmbda=0.95,
        value_network=tensor_dict_module
    )
    # The output tensor dict from GAE will have the following keys.
    gae.set_keys(
        # It's a measurement of how better the action was compared to the estimation.
        advantage="advantage",
        # The "actual" value that the value network should've generated.
        # Note that it's not necessarily the true value.
        # GAE calculates it by taking rewards and future rewards into consideration.
        value_target="value_target",
        # The value from the value network.
        value="value",
    )
    # Input for the GAE.
    # Note that we must use the exact key names shown below except for the "states" key.
    # The "states" key is the one we use for the `tensor_dict_module` above.
    tensor_dict = TensorDict({
        # It refers to the data after taking an action.
        "next": {
            # This will influence the `value_target` because future rewards will be estimated using this.
            "states": torch.FloatTensor([[4, 5], [6, 7]]),
            "reward": torch.FloatTensor([[1], [-1]]),
            # It indicates if the episode has ended.
            # 0 means false, 1 means true.
            "done": torch.BoolTensor([[1], [1]]),
            # The difference between "done" and "terminated" is that done includes termination outside the MDP process.
            # For example, `done` can be true if the episode ends due to time out.
            # The "terminated" only considers the termination by the MDP process (either success or failure).
            "terminated": torch.BoolTensor([[1], [1]])
        },
        "states": torch.FloatTensor([[0, 1], [2, 3]])
    }, batch_size=2)
    output_tensor_dict = gae(tensor_dict)
    print(f"output_tensor_dict: {output_tensor_dict}")
    advantage = output_tensor_dict["advantage"]
    print(f"advantage: {advantage}")

main()
```

Here is the console output:
```
output_tensor_dict: TensorDict(
    fields={
        advantage: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        next: TensorDict(
            fields={
                done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                states: Tensor(shape=torch.Size([2, 2]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                value: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([2]),
            device=None,
            is_shared=False),
        states: Tensor(shape=torch.Size([2, 2]), device=cpu, dtype=torch.float32, is_shared=False),
        value: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        value_target: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([2]),
    device=None,
    is_shared=False)
advantage: tensor([[ 1.4274],
        [-1.8038]])
```
