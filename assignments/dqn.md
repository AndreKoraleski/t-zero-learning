# Assignment: Deep Q-Networks (DQN)

**Due: one week from today.** Work individually or in pairs (state your pair in the report).

You are given a nearly complete DQN implementation ([algorithms/dqn.py](../algorithms/dqn.py)),
adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl) and integrated into this repo's
training harness. The whole algorithm — replay buffer, TD target, epsilon-greedy rollout,
target network — lives in that one file; the framework provides run plumbing (configs,
seeding, logging, run directories). Your job has two parts: a short implementation part,
and an experimental report — **the report is where most of the grade is.**

## Setup

From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Metrics go to [Weights & Biases](https://wandb.ai) (same account you use in other
classes). Create a `.env` file in the repo root:

```
WANDB_PROJECT=dqn-assignment
```

and run `wandb login` once.

## Part 1 & 2 — Implementation (30%)

Complete the three blocks marked `YOUR CODE HERE` in `algorithms/dqn.py`:

- **Part 1a** — `ReplayBuffer.add`: store a transition, FIFO overwrite when full.
- **Part 1b** — `ReplayBuffer.sample`: uniform random minibatch of stored transitions.
- **Part 2** — `compute_td_targets`: the one-step TD target
  $y = r + \gamma \max_{a'} Q_{\text{target}}(s', a') \cdot (1 - \text{done})$.

Each block is a handful of lines. Verify with:

```bash
python -m pytest tests/test_dqn.py
```

All tests must pass before you start Part 3. Then train (always from the repo root):

```bash
python train.py --config dqn_cartpole
```

Hyperparameters live in [configs/dqn_cartpole.yml](../configs/dqn_cartpole.yml); any of
them can be changed per run with `--override` (see below) — you should not need to edit
the config file itself.

A full CartPole run takes roughly 10–20 minutes on a laptop CPU and should reach an
episodic return near 500 (the maximum). If it doesn't, something is wrong — the tests
passing is necessary but not sufficient.

## Part 3 — Experimental report (70%)

For each question below, follow this exact protocol:

1. **Predict** (before running anything): write 2–4 sentences on what you expect to
   happen to the episodic return and TD loss curves, and *why*, based on how the
   algorithm works.
2. **Run**: the baseline and the modified configuration, each with at least 2 seeds
   (`--override seed=1`, `--override seed=2`). Include the wandb plots in the report.
3. **Explain**: 1–2 paragraphs. Did the result match your prediction? Explain the
   *mechanism* behind what you observed — "it got worse" is a description, not an
   explanation.

**Q1 — Target network.** Sync the target network every single step:

```bash
python train.py --config dqn_cartpole --override dqn.target_network_frequency=1
```

What role does the target network play in DQN? What happens to the TD loss when the
target moves at every step, and why can the return curve degrade even while the loss
looks "fine"?

**Q2 — Replay buffer size.** Compare a tiny buffer against the default:

```bash
python train.py --config dqn_cartpole --override dqn.buffer_size=500
```

What two distinct problems does a very small buffer cause? (Hint: think about both the
*correlation* of samples within a minibatch and *what data the network gets to see
again*.)

**Q3 — Your choice.** Pick any one hyperparameter (`dqn.gamma`, `dqn.learning_rate`,
`dqn.exploration_fraction`, `dqn.end_e`, `dqn.batch_size`, `dqn.train_frequency`, ...),
choose a value you expect to change behavior meaningfully, and apply the same
predict → run → explain protocol. Trivial choices (e.g., changing the seed) score zero.

## Grading

| Component | Weight |
|---|---|
| Tests pass + working training run | 30% |
| Q1–Q3: quality of predictions and explanations | 60% |
| Report clarity (plots labeled, configs stated, seeds reported) | 10% |

**Wrong predictions cost nothing.** A wrong prediction followed by a correct mechanistic
explanation of what actually happened gets full marks. A correct prediction with a
hand-wavy explanation does not.

## On AI assistants

You may use them. The implementation blanks are small enough that outsourcing them
saves you little, and the report is graded on reasoning that is yours to defend — be
prepared to explain any line of your code and any paragraph of your report if asked.

## Deliverables

A link to your fork containing your completed `algorithms/dqn.py`, plus a PDF report
(max ~4 pages) with the three predict → run → explain sections.
