# RELAP5 Optimizer

RELAP5 Optimizer is a Python-based optimization and visualization system for tuning RELAP5 control parameters. It combines Bayesian optimization, genetic search, RELAP batch execution, objective calculation, failure detection, and Streamlit dashboards.

The genetic optimizer can run multiple isolated RELAP5 simulations in parallel. The default dev/prod configs use `parallel_workers: 5`, creating worker folders under `runs/parallel_relap`.

## Main Entry Points

- `auto_optimize.py`: main optimization runner.
- `optimization_dashboard.py`: Streamlit optimization dashboard.
- `relap_run_dashboard.py`: RELAP run monitor dashboard.
- `optimization_trace_view.py`: optimization trace analysis view.
- `doc/SYSTEM_DEVELOPMENT_GUIDE.md`: system development guide and file interaction notes.

## Quick Start

Install the required Python packages, then run:

```powershell
python auto_optimize.py --config config_dev.json
streamlit run optimization_dashboard.py
```

The project intentionally excludes local RELAP executables, generated output files, logs, caches, and optimization run artifacts from Git.

## Parallel RELAP Runs

Parallel execution is configured in `integral_optimizer`:

```json
{
  "optimizer": "genetic",
  "parallel_workers": 5,
  "parallel_run_root": "runs/parallel_relap"
}
```

Each worker folder runs its own `start.bat` with its own `indta.i`, `outdta`, `rstplt`, and `plot` outputs, so simulations do not overwrite one another.

## Tests

```powershell
python -m unittest discover -s tests
```
