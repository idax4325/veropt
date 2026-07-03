# Visualisation Improvements

Changes to the model prediction plots and the `choose_plot_point` helper in this release.

---

## 1 — `choose_plot_point`: string-based point selection + new default

**File:** `veropt/graphical/_model_visualisation.py`, lines 22–96

### What changed
- Added `point_selection: Optional[str] = None` parameter.
- **Default behaviour changed**: `None` now always picks the **best evaluated point**
  (weighted objective sum), regardless of whether suggested points are active.
  Previously the default was the *first* suggested point when suggestions existed.
- New string selectors:

| Selector | Behaviour |
|----------|-----------|
| `None` or `"best"` | Best evaluated point (weighted sum of objectives) |
| `"best {objective_name}"` | Best point for a specific named objective |
| `"suggested {n}"` | nth suggested point, 1-indexed |

- `include_suggested_points` kept for backward compatibility (used by
  `plot_prediction_surface_grid` which always evaluates at a known point).

### Code
```python
def choose_plot_point(
        optimiser: BayesianOptimiser,
        normalised: bool,
        include_suggested_points: bool = True,
        point_selection: Optional[str] = None
) -> tuple[torch.Tensor, str]:
    if point_selection is not None and point_selection.startswith("suggested"):
        # parse "suggested N" (1-indexed)
        ...
    elif point_selection is not None and point_selection.startswith("best "):
        # parse "best {objective_name}"
        ...
    else:
        # None or "best" → best evaluated point
        max_ind = optimiser.get_best_points()['index']
        ...
```

---

## 2 — `plot_prediction_surface`: `evaluated_point` now optional + accepts strings

**File:** `veropt/graphical/visualisation.py`

### What changed
- `evaluated_point` parameter type widened to `Optional[Union[torch.Tensor, int, str]]`.
- Now **defaults to `None`** (was previously a required positional argument).
- When a `str` is passed it is forwarded to `choose_plot_point(point_selection=...)`.

### Before / After

```python
# Before: evaluated_point was required
plot_prediction_surface(optimiser, "x1", "x2", "obj1", None)

# After: optional, and accepts strings
plot_prediction_surface(optimiser, "x1", "x2", "obj1")                    # best (default)
plot_prediction_surface(optimiser, "x1", "x2", "obj1", "best")
plot_prediction_surface(optimiser, "x1", "x2", "obj1", "best my_obj")     # best for one obj
plot_prediction_surface(optimiser, "x1", "x2", "obj1", "suggested 2")     # 2nd suggested
plot_prediction_surface(optimiser, "x1", "x2", "obj1", 3)                 # point index 3
```

---

## 3 — `plot_prediction_grid` and `plot_prediction_surface_grid`: same update

Both functions received the identical widening — `evaluated_point` now accepts `str` in
addition to `torch.Tensor`, `int`, and `None`. The string is forwarded to
`choose_plot_point(point_selection=...)`.

`plot_prediction_surface_grid` preserves `include_suggested_points=False` when calling
`choose_plot_point`, so it always evaluates at a known measured point (appropriate since
the surface grid is expensive to compute).

