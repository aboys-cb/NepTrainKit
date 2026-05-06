# Documentation Screenshot System

This tool generates stable UI screenshots for the Sphinx documentation by
starting the real NepTrainKit Qt application, moving it into a named scenario,
capturing the main window, and drawing lightweight callouts.

## Commands

List scenarios:

```bash
python tools/docs/screenshots/capture.py list
```

Update one screenshot:

```bash
python tools/docs/screenshots/capture.py update make_data_lattice_strain
```

Update all screenshots:

```bash
python tools/docs/screenshots/capture.py update --all
```

Check whether committed screenshots are still current:

```bash
python tools/docs/screenshots/capture.py check --all
```

The check command compares rendered pixels with a small tolerance because local
Windows title bars, focus state, and Qt text antialiasing can differ slightly
between captures. Tighten the threshold when debugging one image:

```bash
python tools/docs/screenshots/capture.py check make_data_empty --max-diff-ratio 0.01
```

## Design Rules

- The tool reuses `NepTrainKit.main.create_app()` and
  `NepTrainKit.main.create_main_window()` so screenshots use the same theme,
  font, stylesheet, and window initialization as the real app.
- Scenarios live in `registry.py` and `scenarios.py`.
- Prefer widget/action targets over hard-coded coordinates. Manual boxes are
  only for plot-internal regions that Qt cannot expose as widgets.
- Dataset scenarios copy tracked fixtures from `tests/data/` into
  `.tmp/docs-screenshots/fixtures/` before loading them, so generated analysis
  files never pollute test data.
- Generated images are written to `docs/source/_static/image/generated/`.
- `check` writes fresh images under `.tmp/docs-screenshots/` and compares them
  with the committed images.

## Adding a Scenario

1. Add a `ScenarioSpec` in `registry.py`.
2. Add a runner function in `scenarios.py`.
3. Add annotations with targets such as:

```python
Annotation("1", "Add new card", "widget:make_data_interface.setting_group.new_card_button", "right")
Annotation("2", "Run selected cards", "action:make_data_interface.setting_group:Run", "right")
```

4. Run:

```bash
python tools/docs/screenshots/capture.py update <scenario-name>
```
