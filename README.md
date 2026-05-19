# RELAP5 Optimizer

RELAP5 Optimizer is a Python-based optimization and visualization system for tuning RELAP5 control parameters. It combines Bayesian optimization, genetic search, RELAP batch execution, objective calculation, failure detection, and Streamlit dashboards.

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

## Tests

```powershell
python -m unittest discover -s tests
```

