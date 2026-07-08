# Testing guide

How the test suite is organized, what it guarantees, and what you must add
when you extend the repo. Run everything with:

```bash
pytest                  # full suite (~4s)
pytest -m "not slow"    # fast tier only — config, components, semantics
pytest -m slow          # smoke tier — real (tiny) training runs on CPU
```

## The two kinds of tests

Every test here is one of two kinds. Knowing which is which tells you what
you get for free and what you must write yourself.

### Mechanical (contract) tests — generalize automatically

These check that the *plumbing* works: things load, step, save, resume, and
fail loudly when misused. They are **driven by the repo's registries**, so
new additions are covered without writing any test code:

| Registry you extend | Tests that pick it up automatically |
| --- | --- |
| `configs/*.yml` (a new config file) | `test_configs_load.py` — must load into valid `Args` |
| `ALGORITHMS` in `core/config_loader.py` | `test_train_smoke.py`, `test_resume_smoke.py`, default round-trip in `test_config_roundtrip.py` |
| `WRAPPER_STACKS` in `envs/wrappers.py` | `test_wrappers.py::test_stack_produces_working_env` |
| Env registration in `envs/custom_envs/__init__.py` | `test_custom_envs.py` contract tests (spaces, finite obs/rewards, seed determinism) |

Registering something is *opting in* to its contract tests — you cannot
forget to test the mechanics.

One guard makes this enforcement explicit:
`test_train_smoke.py::test_every_algorithm_has_smoke_settings` fails when an
algorithm is registered without an entry in `tests/helpers.py::SMOKE_SETTINGS`.
That failure is intentional — it is the suite telling you the one manual step
you owe it.

### Semantic tests — deliberately specific, you write them

No generic test can know that AntDirGoal's forward reward is `v · û`, that
it appends 8 goal features, or that the direction switches after step N.
That is the *meaning* of the env, and only its author knows it. Semantic
tests encode that meaning so a silent change (a sign flip, a dropped feature)
becomes a red test instead of an invalid experiment.

Semantic tests do not generalize — **that is the point**. Each env/algorithm/
network gets its own small set, and the existing ones are the template to
copy.

## What you must add when extending the repo

**A new environment** (see also `docs/adding-a-new-environment.md`):
1. Register it in `envs/custom_envs/__init__.py` — contract tests now cover it.
2. Add semantic tests to `tests/test_custom_envs.py`: reward math checked
   against a hand-computed expression from `info`, observation shape/feature
   layout, and any scheduling behavior. Use the `HalfCheetahVel` /
   `AntDirGoal` tests as templates.

**A new algorithm** (e.g. DQN, SAC):
1. Register it in `ALGORITHMS` — the guard test now fails.
2. Add its entry to `SMOKE_SETTINGS` in `tests/helpers.py`: the cheapest env
   with matching space types (e.g. `CartPole-v1` for discrete actions) and a
   `total_timesteps` that is an exact multiple of the rollout/update size.
   Smoke, resume, and config round-trip tests now cover it.
3. If the algorithm has behavior worth pinning beyond the generic mechanics
   (e.g. a replay-buffer invariant), add a small semantic test file for it.

**A new wrapper stack**:
1. Add it to `WRAPPER_STACKS` — the generic contract test now covers it.
2. Add semantic tests to `tests/test_wrappers.py` for what the stack actually
   does (what it clips/normalizes/rejects), following the
   `continuous_control` tests.

**A new network/agent architecture**:
There is no network registry (networks belong to their algorithms), so there
is no automatic coverage beyond what the algorithm's smoke test exercises.
Give the network its own test file modeled on `tests/test_agent_network.py`:
output shapes, config-parameter resolution, seeded-init determinism.

## Rules of thumb

- **Never assert learning performance.** RL is stochastic; "return > X" is a
  flaky test. Assert mechanics (finite losses, files written, steps counted)
  and semantics (reward formulas, feature layouts) instead.
- **Smoke configs are built in code, never committed** — see
  `tests/helpers.make_smoke_args`. `configs/` is for real experiments.
- **Fast tier stays fast.** Anything that runs a training loop gets
  `@pytest.mark.slow` (or module-level `pytestmark = pytest.mark.slow`).
  Component tests that step an env a few times belong in the fast tier.
- **Tests pin decisions.** Unknown config keys and override
  paths are hard errors (fail at startup, never after a 50M-step run);
  `test_overrides.py` and `test_config_roundtrip.py` encode that choice.
- **Hash stability is sacred.** `test_run_naming.py` hardcodes an expected
  hash because run-directory names on disk depend on it. Do not "fix" that
  test by recomputing the hash.

## CI

`.github/workflows/ci.yml` runs the fast tier and then the smoke tier on
every push to `main` and every pull request, on a CPU-only Ubuntu runner
(CPU PyTorch wheel, no rendering, wandb disabled). If the suite passes
locally but fails on CI, the usual suspect is a dependency missing from
`requirements.txt` — CI installs from scratch, your machine does not.
