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
    # Also, the input must be batched.
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

Alternatively, we can use the [generalized_advantage_estimate](https://pytorch.org/rl/stable/reference/generated/torchrl.objectives.value.functional.generalized_advantage_estimate.html) function.

We may want to use that function instead because [GAE does not support an LSTM-based value network](https://github.com/pytorch/rl/issues/2444).

Here is the sample code:
```python
import torch
from torchrl.objectives.value.functional import generalized_advantage_estimate


def main():
    # We must use batched inputs.
    advantage, value_target = generalized_advantage_estimate(
        gamma=0.98,
        lmbda=0.95,
        state_value=torch.FloatTensor([[-1], [0]]),
        next_state_value=torch.FloatTensor([[1], [2]]),
        reward=torch.FloatTensor([[0], [1]]),
        done=torch.BoolTensor([[0], [1]]),
        terminated=torch.BoolTensor([[0], [1]])
    )
    print(f"advantage: {advantage}, value_target: {value_target}")

main()
```

And here is the console output:
```
advantage: tensor([[2.9110],
        [1.0000]]), value_target: tensor([[1.9110],
        [1.0000]])
```

## ClipPPOLoss

```python
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, InteractionType
from torch.distributions import Categorical
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

seed = 3
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def main():
    policy_network = nn.Linear(2, 4)
    policy_module = TensorDictModule(
        module=policy_network,
        in_keys=["observation"],
        out_keys=["logits"]
    )
    actor = ProbabilisticActor(
        module=policy_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        default_interaction_type=InteractionType.MODE,
        # This must be true when using `ClipPPOLoss`.
        return_log_prob=True
    )
    value_network = nn.Linear(2, 1)
    value_operator = ValueOperator(
        module=value_network,
        in_keys=["observation"],
        out_keys=["value"]
    )
    gae = GAE(
        gamma=0.98,
        lmbda=0.95,
        value_network=value_operator
    )
    gae.set_keys(
        advantage="advantage",
        value_target="value_target",
        value="value"
    )

    # PPO
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=value_operator
    )
    # The following keys are required in the input tensor dict for PPO.
    # On top of that, PPO calls the policy and value networks.
    # For that reason, we need to include the parameter keys for them.
    # In this case, they need the "observation" key.
    # Note that we don't set keys for them here because they are not part of "accepted keys" in PPO.
    # You may wonder why PPO still calls the policy and value networks even though their values are already included in the input tensor dict.
    # Apparently, that's how the surrogate objective function works in PPO.
    # It needs to compare the values from the old and current policy to calculate the loss.
    # The purpose is to avoid drastic policy changes and make the training process more stable.
    loss_module.set_keys(
        # Output from `GAE`.
        advantage="advantage",
        # Output from `GAE`.
        value_target="value_target",
        # Output from the value network.
        value="value",
        # Output from `ProbabilisticActor`.
        action="action",
        # Output from `ProbabilisticActor`.
        sample_log_prob="sample_log_prob",
    )

    # Initial input tensor dict.
    # It will be mutated as we apply the policy network, value network, and GAE.
    current_tensor_dict = TensorDict({
        "observation": torch.FloatTensor([[0, 1], [2, 3]])
    }, batch_size=[2])
    # This will append new keys to the input tensor dict to include the output from the actor.
    actor(current_tensor_dict)
    # It prevents the following error: "RuntimeError: tensordict prev_log_prob requires grad."
    current_tensor_dict["sample_log_prob"] = current_tensor_dict["sample_log_prob"].detach()

    # We let this represent the new observation after taking action.
    # We need this for `GAE`.
    next_tensor_dict = TensorDict({
        "observation": torch.FloatTensor([[4, 5], [6, 7]])
    }, batch_size=[2])

    # Connect the current and next tensor dict for `GAE`.
    current_tensor_dict["next"] = next_tensor_dict
    next_tensor_dict["reward"] = torch.FloatTensor([[1], [-1]])
    next_tensor_dict["done"] = torch.BoolTensor([[1], [1]])
    next_tensor_dict["terminated"] = torch.BoolTensor([[1], [1]])

    # This will append new keys to the input tensor dict to include the output from `GAE`.
    # Actually, we don't have to call this because PPO will internally use `GAE` if we don't call it.
    # By the way, we can use custom logic to calculate `advantage` and `value_target` and directly append them into the input tensor dict.
    # It can be useful knowledge, especially when we cannot use `GAE` due to its limitations.
    # For example, `GAE` does not support an LSTM-based value network.
    gae(current_tensor_dict)

    # This shows the mutated result.
    print(f"current_tensor_dict: {current_tensor_dict}")

    loss_tensor_dict = loss_module(current_tensor_dict)
    print(f"loss_tensor_dict: {loss_tensor_dict}")
    loss_critic = loss_tensor_dict["loss_critic"]
    loss_entropy = loss_tensor_dict["loss_entropy"]
    loss_objective = loss_tensor_dict["loss_objective"]
    loss = loss_critic + loss_entropy + loss_objective
    print(f"loss: {loss}")

main()
```

Here is the console output:
```
current_tensor_dict: TensorDict(
    fields={
        action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.int64, is_shared=False),
        advantage: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        logits: Tensor(shape=torch.Size([2, 4]), device=cpu, dtype=torch.float32, is_shared=False),
        next: TensorDict(
            fields={
                done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                observation: Tensor(shape=torch.Size([2, 2]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                value: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([2]),
            device=None,
            is_shared=False),
        observation: Tensor(shape=torch.Size([2, 2]), device=cpu, dtype=torch.float32, is_shared=False),
        sample_log_prob: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
        value: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        value_target: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([2]),
    device=None,
    is_shared=False)
loss_tensor_dict: TensorDict(
    fields={
        ESS: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
        clip_fraction: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
        entropy: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
        kl_approx: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
        loss_critic: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
        loss_entropy: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
        loss_objective: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([]),
    device=None,
    is_shared=False)
loss: -0.5069577097892761
```

Here is another example for multiple episodes:
```python
import torch
from tensordict import TensorDict
from torch import nn
from tensordict.nn import TensorDictModule, InteractionType
from torch.distributions import Categorical
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE


def main():
    policy_network = nn.Linear(2, 4)
    policy_module = TensorDictModule(
        module=policy_network,
        in_keys=["states"],
        out_keys=["logits"]
    )
    actor = ProbabilisticActor(
        module=policy_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        default_interaction_type=InteractionType.MODE,
        return_log_prob=True
    )
    value_network = nn.Linear(2, 1)
    value_operator = ValueOperator(
        module=value_network,
        in_keys=["states"],
        out_keys=["values"]
    )
    gae = GAE(
        gamma=0.98,
        lmbda=0.95,
        value_network=value_operator
    )
    gae.set_keys(
        value="values"
    )

    # PPO
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=value_operator
    )
    loss_module.set_keys(
        value="values"
    )

    # Input tensor dicts
    current_tensor_dict = TensorDict({
        "states": torch.FloatTensor([[0, 1]])
    }, batch_size=1)
    next_tensor_dict = TensorDict({
        "states": torch.FloatTensor([[2, 3]])
    }, batch_size=1)
    next_next_tensor_dict = TensorDict({
        "states": torch.FloatTensor([[4, 5]])
    }, batch_size=1)

    actor(current_tensor_dict)
    actor(next_tensor_dict)
    actor(next_next_tensor_dict)

    current_tensor_dict["sample_log_prob"] = current_tensor_dict["sample_log_prob"].detach()
    next_tensor_dict["sample_log_prob"] = next_tensor_dict["sample_log_prob"].detach()

    # First episode.
    current_tensor_dict["next"] = next_tensor_dict
    next_tensor_dict["reward"] = torch.FloatTensor([[0]])
    next_tensor_dict["done"] = torch.BoolTensor([[0]])
    next_tensor_dict["terminated"] = torch.BoolTensor([[0]])

    # Second episode.
    next_tensor_dict["next"] = next_next_tensor_dict
    next_next_tensor_dict["reward"] = torch.FloatTensor([[-10]])
    next_next_tensor_dict["done"] = torch.BoolTensor([[1]])
    next_next_tensor_dict["terminated"] = torch.BoolTensor([[1]])

    # We must call the `value_operators` for all states if we have multiple episodes.
    # Otherwise, GAE will throw exceptions.
    value_operator(current_tensor_dict)
    value_operator(next_tensor_dict)
    value_operator(next_next_tensor_dict)

    # Calculate advantages for the first and second episodes.
    gae(current_tensor_dict)
    gae(next_tensor_dict)

    # We need to call the `loss_module` for each episode.
    loss_current_tensor_dict = loss_module(current_tensor_dict)
    print(f"loss_current_tensor_dict: {loss_current_tensor_dict}")
    loss_critic = loss_current_tensor_dict["loss_critic"]
    loss_entropy = loss_current_tensor_dict["loss_entropy"]
    loss_objective = loss_current_tensor_dict["loss_objective"]
    loss = loss_critic + loss_entropy + loss_objective
    print(f"loss: {loss}")

    # Loss for the second episode.
    loss_next_tensor_dict = loss_module(next_tensor_dict)
    print(f"loss_next_tensor_dict: {loss_next_tensor_dict}")
    loss_critic = loss_next_tensor_dict["loss_critic"]
    loss_entropy = loss_next_tensor_dict["loss_entropy"]
    loss_objective = loss_next_tensor_dict["loss_objective"]
    loss = loss_critic + loss_entropy + loss_objective
    print(f"loss: {loss}")

main()
```

## Exploration in Training

```python
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, InteractionType
from torch import optim
from torch.distributions import Categorical
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.functional import generalized_advantage_estimate


def main():
    seed = 3
    torch.manual_seed(seed)

    # Find the maximum number in the list of 10 numbers.
    policy_network = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    policy_module = TensorDictModule(
        module=policy_network,
        in_keys=["numbers"],
        out_keys=["logits"]
    )
    actor = ProbabilisticActor(
        module=policy_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        # For training, it's better to randomly pick the action from the distribution to increase the exploration.
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=True
    )
    value_network = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    value_operator = ValueOperator(
        module=value_network,
        in_keys=["numbers"],
        out_keys=["value"]
    )
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=value_operator
    )
    loss_module.set_keys(
        advantage="advantage",
        value_target="value_target",
        value="value",
        action="action",
        sample_log_prob="sample_log_prob"
    )

    # Training

    epoch = 0
    max_epoch = 5000
    optimizer = optim.Adam(loss_module.parameters())
    number_of_correct_decisions = 0
    total_loss = 0
    numbers = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    while epoch < max_epoch:
        current_tensor_dict = TensorDict({
            "numbers": torch.FloatTensor(numbers)
        }, batch_size=[])
        actor(current_tensor_dict)
        max_index = current_tensor_dict["action"].item()
        value_operator(current_tensor_dict)
        current_tensor_dict["sample_log_prob"] = current_tensor_dict["sample_log_prob"].detach()
        next_tensor_dict = TensorDict({
            "numbers": torch.FloatTensor(numbers)
        }, batch_size=[])
        value_operator(next_tensor_dict)

        correct_index = numbers.index(max(numbers))

        # Reward/Penalty Rules
        score = 0
        if max_index == correct_index:
            score += 10
            number_of_correct_decisions += 1
        else:
            score -= 1
        reward = torch.FloatTensor([[score]])

        # Note that we need to use batched input, and the output will be in batched form.
        advantage, value_target = generalized_advantage_estimate(
            gamma=0.98,
            lmbda=0.95,
            state_value=current_tensor_dict["value"].unsqueeze(0),
            next_state_value=next_tensor_dict["value"].unsqueeze(0),
            reward=reward,
            done=torch.BoolTensor([[1]]),
            terminated=torch.BoolTensor([[1]])
        )
        current_tensor_dict["advantage"] = advantage.squeeze(0)
        current_tensor_dict["value_target"] = value_target.squeeze(0)
        loss_tensor_dict = loss_module(current_tensor_dict)
        loss_critic = loss_tensor_dict["loss_critic"]
        loss_entropy = loss_tensor_dict["loss_entropy"]
        loss_objective = loss_tensor_dict["loss_objective"]
        loss = loss_critic + loss_entropy + loss_objective
        total_loss += loss
        print(f"episode: {epoch}, score: {score}, max_num: {numbers[max_index]}")
        # It's important to let the agent explore the environment enough so that it can experience the rewards.
        # That's why we optimize the models once every 100 epochs.
        # More about how rewards affect the training: https://stackoverflow.com/a/79016701/11717859
        if epoch % 100 == 0:
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss = 0
        epoch += 1
    print(f"Accuracy = {number_of_correct_decisions / max_epoch}")

main()
```

## Train Transformer

```python
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, InteractionType
from torch import optim
from torch.distributions import Categorical
from torch.nn import Embedding
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.functional import generalized_advantage_estimate


class OurEmbedding:

    def __init__(self, device):
        self.token_to_index = {
            "<sos>": 0,
            "<eos>": 1,
            "roses": 2,
            "are": 3,
            "red": 4
        }
        self.vocab_size = len(self.token_to_index)
        self.token_vec_dim = 8
        self.device = device
        self.torch_embedding = Embedding(self.vocab_size, self.token_vec_dim, device=self.device)


class PolicyNetwork(nn.Module):

    def __init__(self, our_embedding, device):
        super().__init__()
        self.our_embedding = our_embedding
        self.device = device
        self.transformer = nn.Transformer(
            d_model=self.our_embedding.token_vec_dim,
            nhead = 8 if self.our_embedding.token_vec_dim % 8 == 0 else 2,
            batch_first = True,
            device=self.device
        )
        self.fc = nn.Sequential(
            nn.Linear(self.our_embedding.token_vec_dim, self.our_embedding.token_vec_dim * 2, device=self.device),
            nn.ReLU(),
            nn.Linear(self.our_embedding.token_vec_dim * 2, self.our_embedding.vocab_size, device=self.device)
        )

    def forward(self, src_token_vecs, target_token_vecs):
        """
        Predict the next token.
        Note that we assume the inputs are not batched.
        :param src_token_vecs: Same meaning as the `nn.Transformer.forward(src, tgt)`.
                               Shape: (number_of_tokens, token_vec_dim).
        :param target_token_vecs: Same meaning as the `nn.Transformer.forward(src, tgt)`.
                                  Shape: (number_of_tokens, token_vec_dim).
        :return: The raw score (logit) for the next token.
                 Shape: (vocab_size).
        """
        # Shape: (number_of_tokens, token_vec_dim).
        transformer_output = self.transformer(src_token_vecs, target_token_vecs)
        # Shape: (token_vec_dim).
        last_token_vec = transformer_output[-1]
        return self.fc(last_token_vec)

class ValueNetwork(nn.Module):

    def __init__(self, our_embedding, device):
        super().__init__()
        self.our_embedding = our_embedding
        self.device = device
        self.transformer = nn.Transformer(
            d_model=self.our_embedding.token_vec_dim,
            nhead=8 if self.our_embedding.token_vec_dim % 8 == 0 else 2,
            batch_first=True,
            device=self.device
        )
        self.fc = nn.Sequential(
            nn.Linear(self.our_embedding.token_vec_dim, self.our_embedding.token_vec_dim * 2, device=self.device),
            nn.ReLU(),
            nn.Linear(self.our_embedding.token_vec_dim * 2, 1, device=self.device)
        )

    def forward(self, src_token_vecs, generated_token_vecs):
        """
        Evaluate the performance of the policy.
        Note that we assume that the inputs are not batched.
        :param src_token_vecs: Same meaning as the `nn.Transformer.forward(src, tgt)`.
                               Shape: (number_of_tokens, token_vec_dim).
        :param generated_token_vecs: Shape: (number_of_tokens, token_vec_dim).
        :return: Value. Shape: (1).
        """
        # Shape: (number_of_tokens, token_vec_dim).
        transformer_output = self.transformer(src_token_vecs, generated_token_vecs)
        last_element = transformer_output[-1]
        return self.fc(last_element)


def main():
    torch.autograd.set_detect_anomaly(True)
    seed = 3
    device = torch.device("cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(seed)

    our_embedding = OurEmbedding(device)
    policy_network = PolicyNetwork(our_embedding, device)
    policy_module = TensorDictModule(
        module=policy_network,
        in_keys=["src_token_vecs", "target_token_vecs"],
        out_keys=["logits"]
    )
    actor = ProbabilisticActor(
        module=policy_module,
        in_keys=["logits"],
        # Shape: a scalar representing the next token's index.
        out_keys=["action"],
        distribution_class=Categorical,
        # Make decision in a stochastic manner.
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=True
    )
    value_network = ValueNetwork(our_embedding, device)
    value_operator = ValueOperator(
        module=value_network,
        in_keys=["src_token_vecs", "target_token_vecs"],
        out_keys=["value"]
    )
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=value_operator
    )
    loss_module.set_keys(
        advantage="advantage",
        value_target="value_target",
        value="value",
        action="action",
        sample_log_prob="sample_log_prob"
    )

    # Training

    epoch = 0
    max_epoch = 10000
    optimizer = optim.Adam(loss_module.parameters())
    total_loss = 0
    while epoch < max_epoch:
        # We start with 1 because the initial "<sos>" token is already used.
        token_gen_iteration = 1
        # It will generate 10 tokens at maximum.
        max_token_gen_iteration = 10
        # Shape: (number_of_tokens, token_vec_dim).
        src_token_vecs = our_embedding.torch_embedding(torch.LongTensor([2, 3]).to(device))
        # Assuming index=0 corresponds to the "<sos>" token.
        target_token_indexes = [0]
        # It's mutable.
        tensor_dict = TensorDict({}, batch_size=[])
        current_tensor_dict = tensor_dict
        # Take actions until we get the "<eos>" token or reach the maximum token lengths.
        while True:
            # Shape: (number_of_tokens, token_vec_dim).
            target_token_vecs = our_embedding.torch_embedding(torch.LongTensor(target_token_indexes).to(device))
            current_tensor_dict["src_token_vecs"] = src_token_vecs
            current_tensor_dict["target_token_vecs"] = target_token_vecs
            value_operator(current_tensor_dict)
            # Assuming index=1 corresponds to the "<eos>" token.
            if target_token_indexes[-1] == 1 or token_gen_iteration >= max_token_gen_iteration:
                break
            actor(current_tensor_dict)
            current_tensor_dict["sample_log_prob"] = current_tensor_dict["sample_log_prob"].detach()
            # Shape: a scalar.
            next_token_index = current_tensor_dict["action"]
            target_token_indexes.append(next_token_index.item())
            current_tensor_dict["next"] = TensorDict({}, batch_size=[])
            current_tensor_dict = current_tensor_dict["next"]
            token_gen_iteration += 1
        # Iterate through the tensor_dict and calculate the advantages.
        current_tensor_dict = tensor_dict
        while "next" in current_tensor_dict:
            done = torch.BoolTensor([[0]]).to(device)
            terminated = torch.BoolTensor([[0]]).to(device)
            if current_tensor_dict["action"].item() == 1:
                done = torch.BoolTensor([[1]]).to(device)
                terminated = torch.BoolTensor([[1]]).to(device)

            # Reward/Penalty Rules
            score = 0
            if len(target_token_indexes) > 1 and target_token_indexes[1] == 4:
                score += 3
            if len(target_token_indexes) == 3:
                score += 5
            if target_token_indexes == [0, 4, 1]:
                score += 10

            next_tensor_dict = current_tensor_dict["next"]
            # Note that we need to use batched input and the output will be in batched form.
            advantage, value_target = generalized_advantage_estimate(
                gamma=0.98,
                lmbda=0.95,
                state_value=current_tensor_dict["value"].unsqueeze(0),
                next_state_value=next_tensor_dict["value"].unsqueeze(0),
                reward=torch.FloatTensor([[score]]).to(device),
                done=done,
                terminated=terminated
            )
            current_tensor_dict["advantage"] = advantage.squeeze(0)
            current_tensor_dict["value_target"] = value_target.squeeze(0)
            current_tensor_dict = current_tensor_dict["next"]
        # Iterate through the tensor_dict and calculate each loss.
        current_tensor_dict = tensor_dict
        while "next" in current_tensor_dict:
            loss_tensor_dict = loss_module(current_tensor_dict)
            loss_critic = loss_tensor_dict["loss_critic"]
            loss_entropy = loss_tensor_dict["loss_entropy"]
            loss_objective = loss_tensor_dict["loss_objective"]
            loss = loss_critic + loss_entropy + loss_objective
            total_loss += loss
            current_tensor_dict = current_tensor_dict["next"]
        if epoch % 100 == 0:
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss = 0
        print(f"epoch: {epoch}, target_token_indexes: {target_token_indexes}")
        epoch += 1

main()
```
