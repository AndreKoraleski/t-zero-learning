# Adding Metrics

Where metrics live, how they reach the console and Weights & Biases, and
recipes for adding your own. Reference implementation:
`_log_iteration_metrics` in
[algorithms/ppo_continuous_action.py](../algorithms/ppo_continuous_action.py).

## Who owns what

Logging is split the same way as everything else in this repo:

- **The base class owns the infrastructure**: `wandb.init` and the
  `special_log_every` schedule counters come from
  `_setup_logging_and_checkpoints()` in
  [algorithms/base.py](../algorithms/base.py).
- **Each algorithm owns the content.** PPO has a method called `_log_iteration_metrics` which receives batch data and computes metrics for logging. This is NOT part of the `Algorithm` base class, but we encourage new algorithms to follow the example for ease of alteration.

What carries over to a new algorithm is not a shared function but the
**conventions** in this document: the two sinks, the namespaces, the
one-log-call-per-iteration rule, and the slow-cadence pattern.

## How metrics flow

There are exactly two sinks — the tqdm progress bar and wandb (when
`track: true`). There is no TensorBoard writer. In PPO this looks like:

```
train() loop                          _log_iteration_metrics(...)
┌─────────────────────────────┐    ┌────────────────────────────────────────┐
│ rollout: episode returns /  │    │ 1. tqdm postfix: sps, gs, lr, r_roll   │
│   successes from env infos  │ →  │ 2. if track: build metrics_dict and    │
│ update: losses, KL, clip-   │    │    wandb.log(metrics_dict,             │
│   fracs, grad norms, …      │    │              step=global_step)         │
└─────────────────────────────┘    │    — one call per iteration            │
                                   └────────────────────────────────────────┘
```

Design rules worth keeping:

1. **Diagnostics never affect training.** The logging function is called
   once per iteration, after the update; everything inside it is read-only.
   Adding a metric must never change what the optimizer sees.
2. **One `wandb.log` call per iteration, always at `step=global_step`.**
   wandb requires monotonically increasing steps; scattering `wandb.log`
   calls with mixed step values silently drops data.
3. **Collect first, upload once.** `wandb.log` accepts a dictionary — since
   there is only one call per iteration, build the metrics up in a dict
   (`metrics_dict` in PPO) and log it at the end.

## Recipe 1 — a scalar from the update loop

Example: log the mean advantage magnitude.

1. Collect it in your algorithm's `train()` next to the other per-iteration
   temporaries (in PPO: `clipfracs`, `grad_norms`), and pass it to the
   logging function as a keyword argument.
2. Add the matching parameter and one line in the logging function:

```python
metrics_dict["losses/mean_abs_advantage"] = float(b_advantages.abs().mean().item())
```

> Call `.item()` / cast to `float` — logging live tensors keeps graphs and GPU
> memory alive.

## Recipe 2 — a metric from the environment

Per-episode metrics arrive through the terminal `info` dict of the vector
env, extracted by
[envs/custom_envs/envs_utils.py](../envs/custom_envs/envs_utils.py)`::episode_completions_from_vector_infos`,
which handles the two Gymnasium autoreset layouts (`final_info` vs root
`_episode`). It already returns per-episode `success` when the env provides
that key — this is how `rollout/mean_success_rate` appears for Meta-World
but not for MuJoCo envs.

- **Your env exposes a scalar in its terminal info** (e.g. `distance_to_goal`):
  extend `episode_completions_from_vector_infos` to extract it (mirroring the
  `success` handling), accumulate it in the rollout lists in `train()`, and
  log it under `rollout/`.
- **Step-level env quantity** (not per-episode): accumulate it in the rollout
  loop yourself and log an aggregate. Resist logging per-step values — at
  50M steps that's 50M points wandb has to swallow.

## Recipe 3 — expensive metrics on a slow cadence

Weight histograms in PPO show the pattern (end of `_log_iteration_metrics`):

```python
if self.global_step >= self.next_special_log_step:
    for name, param in self.agent.named_parameters():
        metrics_dict[f"weights/{name}"] = wandb.Histogram(
            param.detach().clone().cpu().numpy()
        )
    while self.next_special_log_step <= self.global_step:
        self.next_special_log_step += self.special_log_every
```

We call *special logs* the metrics that are too costly to collect or store
every iteration: weight histograms, whole-network activations, and the
like. Notes:

- The schedule is a **threshold** — `global_step` advances by
  `num_envs` per env step and by a whole batch per iteration, so exact
  multiples are rarely hit. Missed thresholds are skipped, not queued.
- The cadence is configurable per run (`special_log_every`, floored at one
  batch), and `next_special_log_step` should be part of the checkpoint payload. A resumed run should continue the schedule instead of restarting it.

## Recipe 4 — evaluation metrics

Post-training evaluation returns
`{"episodic_returns": [...], "metrics": {"eval/...": ...}}` from
`evaluate_checkpoint` in [algorithms/base.py](../algorithms/base.py); the
`metrics` dict is logged to wandb as-is by `_post_training_eval`. To add an
eval metric, extend that returned dict. Envs with their own evaluation
protocol do the same inside their adapter's `evaluate`
([envs/adapters/](../envs/adapters/)).

## Conventions checklist

- [ ] Pick the right namespace: `losses/` (optimization), `charts/`
      (run-level curves), `rollout/` (per-rollout env outcomes), `weights/`
      (slow-cadence introspection), `eval/` (post-training)
- [ ] Log Python floats/ints, not tensors
- [ ] Add to the iteration's metrics dict, don't call `wandb.log` yourself —
      one call per iteration, `step=global_step`
- [ ] Conditional metrics: only add the key when there is data
      (`rollout/*` keys are absent, not zero, on iterations without
      completed episodes)
- [ ] If your metric needs a counter that must survive resume, put it on
      `self` and add it to `checkpoint_state_dict()` /
      `load_checkpoint_state_dict()` (see
      [checkpoint-semantics.md](checkpoint-semantics.md))
- [ ] If the algorithm has a mirrored variant (like the two PPO files),
      mirror the change there too
