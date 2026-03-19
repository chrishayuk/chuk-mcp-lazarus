# Experiment Persistence

Create and persist named experiments with multiple result steps.

---

## `create_experiment`

```python
async def create_experiment(
    name: str,
    description: str = "",
    metadata: dict | None = None,
) -> dict
```

Create a named experiment in `ExperimentStore`.  Returns an `experiment_id` (UUID).

---

## `add_experiment_result`

```python
async def add_experiment_result(
    experiment_id: str,
    step_name: str,
    result: dict,
    notes: str = "",
) -> dict
```

Append a step result to an experiment.  `result` can be any JSON-serializable dict — typically the output of another tool call.

---

## `get_experiment`

```python
async def get_experiment(experiment_id: str) -> dict
```

Retrieve an experiment and all its result steps.

---

## `list_experiments`

```python
async def list_experiments() -> dict
```

List all saved experiments with metadata.

---

### Usage pattern

Experiments are designed for Claude to record its own reasoning and findings during an interpretability session:

```python
# Create
exp = await create_experiment(
    name="language-transition-gemma-3-4b",
    description="Find crossover layer and steer French → German",
    metadata={"model": "google/gemma-3-4b-it", "date": "2026-03-19"},
)
exp_id = exp["experiment_id"]

# Record each step
await add_experiment_result(exp_id, "probe_scan", scan_result,
    notes="Crossover at layer 18 with 94% val accuracy")
await add_experiment_result(exp_id, "steering", steer_result,
    notes="alpha=20 redirected French→German successfully")

# Review later
full_exp = await get_experiment(exp_id)
```

Experiments persist in memory across tool calls within the same server session.  For cross-session persistence, results are also written to disk.
