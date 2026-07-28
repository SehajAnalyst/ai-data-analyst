# ml_plugins/classification/

Placeholder for the **classification** ML plugin — not implemented in this
skeleton. Per the architecture roadmap (Phase 4), ML plugins are a
separate, project-sized effort built AFTER the core NL2SQL pipeline is
proven, and the first plugin built should validate the
`BaseMLPlugin` interface (ml_plugins/base_plugin.py) end-to-end
before the remaining four are attempted — building all five
speculatively before validating the pattern works is explicitly the
wrong order, per the architecture discussion.

## Planned contents (implementation phase, when this plugin is built)

- `plugin.py` — concrete `BaseMLPlugin` implementation
- `model.py` — the underlying model logic (e.g. scikit-learn usage)
- `param_elicitation.py` — for plugins needing conversational
  parameter gathering (e.g. Forecasting needs to know which time
  column, which horizon) — NOT every plugin needs this file; simpler
  plugins may not require elicitation beyond what
  `validate_requirements()` already surfaces.
