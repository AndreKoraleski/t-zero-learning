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

## Part 1 & 2 — Implementation

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

## The charts you will work with

Every run logs these to wandb; your report should read from them (screenshots or
wandb report links, always with the runs labeled):

| Chart | What it tells you |
|---|---|
| `charts/episodic_return_mean_last100` | the learning curve — mean return over the last 100 episodes |
| `losses/td_loss` | how far Q(s,a) is from the TD target on sampled batches |
| `losses/q_values` | mean predicted Q — watch for divergence or runaway growth |
| `charts/epsilon` | the exploration schedule actually used |
| `charts/episodic_length_mean_last100` | episode length (on CartPole, ≡ return) |
| `eval/mean_return`, `eval/std_return` | final 10-episode greedy evaluation |

A key habit this assignment trains: **never read a single chart in isolation.** The
return curve tells you *whether* something went wrong; `td_loss` and `q_values`
together often tell you *what*.

## Part 3 — Experimental report

For each question below, follow this protocol:

1. **Sweep**: pick **at least three values** of the hyperparameter (your choice,
   spanning small → large enough to expose the behavior), plus the baseline. Run each
   configuration; where feasible, run 2 seeds for the configurations your argument
   hinges on (`--override seed=2`).
2. **Report the charts**: for each question, include the charts named below, with all
   sweep values overlaid, runs labeled.
3. **Explain**: 1–2 paragraphs on the *mechanism* behind what you observed — "it got
   worse" is a description, not an explanation. Explanations must reference the
   charts ("the q_values chart shows …, which means …").

Overrides work like this (any key in [configs/dqn_cartpole.yml](../configs/dqn_cartpole.yml)):

```bash
python train.py --config dqn_cartpole --override dqn.target_network_frequency=1 seed=2
```

**Q1 — Target network sync frequency** (`dqn.target_network_frequency`; 1 = a new
target every step, large = a nearly frozen target).
Charts to report: `charts/episodic_return_mean_last100`, `losses/td_loss`,
`losses/q_values`.
What role does the target network play? Why can the return curve degrade while
`td_loss` still looks "fine"? What does `losses/q_values` do at your extreme values,
and why?

**Q2 — Replay buffer size** (`dqn.buffer_size`; try tiny through generous).
Charts to report: `charts/episodic_return_mean_last100`, `losses/q_values`.
What two distinct problems does a very small buffer cause? (Hint: think about both
the *correlation* of samples within a minibatch and *what data the network gets to
see again*.)

**Q3 — Your choice.** Pick any one other hyperparameter (`dqn.gamma`,
`dqn.learning_rate`, `dqn.exploration_fraction`, `dqn.end_e`, `dqn.batch_size`,
`dqn.train_frequency`, ...), sweep it the same way, and state *which charts you chose
to report and why* — chart selection is part of the answer here (e.g., an exploration
sweep without `charts/epsilon` is incomplete). Trivial choices (changing the seed)
score zero.

## Grading

Exact weights will be announced separately; what is graded:

- tests passing + a working baseline training run;
- Q1–Q3: quality of the sweeps (sensible value ranges) and, above all, of the
  mechanistic explanations grounded in the charts;
- report clarity: plots labeled, configs and seeds stated, runs reproducible from
  what you wrote.

A modest sweep explained well beats an exhaustive sweep described vaguely.

## On AI assistants

You may use them. The implementation blanks are small enough that outsourcing them
saves you little, and the report is graded on reasoning that is yours to defend — be
prepared to explain any line of your code and any paragraph of your report if asked.

## Deliverables

A link to your fork containing your completed `algorithms/dqn.py`, plus a PDF report
(max ~4 pages) with the three sweep → charts → explain sections.
