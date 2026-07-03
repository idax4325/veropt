import json
import os
from pathlib import Path

import pytest

from veropt import bayesian_optimiser
from veropt.optimiser.optimiser_saver_loader import (
    CURRENT_SCHEMA_VERSION,
    _migrate_v1_to_v2,
    migrate_json,
    save_to_json,
    load_optimiser_from_state,
)
from veropt.optimiser.optimiser import BayesianOptimiser
from veropt.optimiser.practice_objectives import Hartmann


def _make_minimal_optimiser(n_initial_points: int = 4, verbose: bool = False) -> BayesianOptimiser:
    """Create a small optimiser for use in saver/loader tests."""
    return bayesian_optimiser(
        n_initial_points=n_initial_points,
        n_bayesian_points=8,
        n_evaluations_per_step=2,
        objective=Hartmann(n_variables=3),
        verbose=verbose,
        model={'training_settings': {'max_iter': 10}},
        acquisition_optimiser={'optimiser': 'dual_annealing', 'optimiser_settings': {'max_iter': 10}}
    )


def _build_v1_saved_dict_fragment() -> dict:
    """Return the minimal dict structure representing a v1 JSON (no schema_version, noise in settings)."""
    return {
        # No 'schema_version' key
        'optimiser': {
            'predictor': {
                'state': {
                    'model': {
                        'state': {
                            'model_dicts': {
                                'model_0': {
                                    'state': {
                                        'settings': {
                                            'lengthscale_lower_bound': 0.1,
                                            'lengthscale_upper_bound': 2.0,
                                            'nu': 2.5,
                                            'noise': 1e-8,
                                            'noise_lower_bound': 1e-8,
                                            'train_noise': False,
                                        },
                                        'state_dict': {},
                                        'train_inputs': [],
                                        'train_targets': [],
                                        'n_variables': 3,
                                    }
                                }
                            }
                        }
                    }
                }
            },
            'settings': {
                'n_initial_points': 4,
                'n_bayesian_points': 8,
                'n_objectives': 1,
                'n_evaluations_per_step': 2,
                'allow_automatic_json_updates': False,
            }
        }
    }


def _downgrade_to_v1(saved: dict) -> dict:
    """Move noise fields back into kernel settings and remove schema_version (simulates old save)."""
    model_dicts = saved['optimiser']['predictor']['state']['model']['state']['model_dicts']
    for model_key in model_dicts:
        state = model_dicts[model_key]['state']
        noise = state.pop('noise', 1e-8)
        noise_lower_bound = state.pop('noise_lower_bound', 1e-8)
        train_noise = state.pop('train_noise', False)
        state['settings']['noise'] = noise
        state['settings']['noise_lower_bound'] = noise_lower_bound
        state['settings']['train_noise'] = train_noise
    saved.pop('schema_version', None)
    return saved


# --- Schema version tests ---

def test_save_produces_current_schema_version(tmp_path: Path) -> None:
    """Saving an optimiser must stamp schema_version in the file."""

    optimiser = _make_minimal_optimiser()
    save_path = str(tmp_path / "optimiser.json")
    save_to_json(optimiser, save_path)

    with open(save_path, 'r') as json_file:
        saved = json.load(json_file)

    assert saved.get('schema_version') == CURRENT_SCHEMA_VERSION


def test_noise_fields_not_in_model_state(tmp_path: Path) -> None:
    """In v4, noise and noise_lower_bound must NOT appear in model/kernel state.
    train_noise is still present (controls whether the GP trains or freezes noise).
    The physical noise values (noise_std etc.) live exclusively on the objective."""

    optimiser = _make_minimal_optimiser()
    save_path = str(tmp_path / "optimiser.json")
    save_to_json(optimiser, save_path)

    with open(save_path, 'r') as json_file:
        saved = json.load(json_file)

    model_dicts = (
        saved['optimiser']['predictor']['state']['model']['state']['model_dicts']
    )

    removed_noise_keys = {'noise', 'noise_lower_bound'}

    for model_key, model_dict in model_dicts.items():
        state = model_dict['state']
        state_keys = set(state.keys())
        settings_keys = set(state.get('settings', {}).keys())

        assert not (removed_noise_keys & state_keys), (
            f"Noise keys {removed_noise_keys & state_keys} found at model state top-level for {model_key} — "
            "they should only exist on the objective in v4."
        )
        assert not (removed_noise_keys & settings_keys), (
            f"Noise keys {removed_noise_keys & settings_keys} still found inside 'settings' for {model_key}."
        )


def test_round_trip_save_load_preserves_train_noise(tmp_path: Path) -> None:
    """Save → load must preserve the train_noise flag on the objective."""

    objective = Hartmann(n_variables=3, train_noise=True)
    optimiser = bayesian_optimiser(
        n_initial_points=4,
        n_bayesian_points=8,
        n_evaluations_per_step=2,
        objective=objective,
        verbose=False,
        model={'training_settings': {'max_iter': 10}},
        acquisition_optimiser={'optimiser': 'dual_annealing', 'optimiser_settings': {'max_iter': 10}}
    )

    save_path = str(tmp_path / "optimiser.json")
    save_to_json(optimiser, save_path)

    loaded_optimiser = load_optimiser_from_state(save_path)

    assert loaded_optimiser.objective.train_noise is True


# --- Migration unit tests ---

def test_migrate_v1_to_v2_moves_noise_fields() -> None:
    """_migrate_v1_to_v2 must move noise fields from settings to state top-level."""

    v1_dict = _build_v1_saved_dict_fragment()
    v2_dict = _migrate_v1_to_v2(v1_dict)

    model_0_state = (
        v2_dict['optimiser']['predictor']['state']['model']['state']['model_dicts']['model_0']['state']
    )

    assert 'noise' in model_0_state
    assert 'noise_lower_bound' in model_0_state
    assert 'train_noise' in model_0_state

    assert 'noise' not in model_0_state['settings']
    assert 'train_noise' not in model_0_state['settings']

    assert v2_dict['schema_version'] == CURRENT_SCHEMA_VERSION


def test_migrate_json_creates_backup_and_updates_file(tmp_path: Path) -> None:
    """migrate_json must create a .bak file and write a v2 file in place."""

    optimiser = _make_minimal_optimiser()
    save_path = str(tmp_path / "optimiser.json")
    save_to_json(optimiser, save_path)

    with open(save_path, 'r') as json_file:
        saved = json.load(json_file)

    saved = _downgrade_to_v1(saved)

    with open(save_path, 'w') as json_file:
        json.dump(saved, json_file)

    migrate_json(save_path)

    backup_path = save_path + '.bak'
    assert os.path.exists(backup_path), "Backup file was not created."

    with open(save_path, 'r') as json_file:
        migrated = json.load(json_file)

    assert migrated.get('schema_version') == CURRENT_SCHEMA_VERSION

    model_0_state = (
        migrated['optimiser']['predictor']['state']['model']['state']['model_dicts']['model_0']['state']
    )
    # In v4, noise and noise_lower_bound are removed from model state; noise_std lives on the objective.
    assert 'noise' not in model_0_state
    assert 'noise_lower_bound' not in model_0_state
    assert 'noise' not in model_0_state.get('settings', {})


def test_load_v1_json_raises_without_allow_flag(tmp_path: Path) -> None:
    """Loading a v1 JSON when allow_automatic_json_updates=False must raise RuntimeError."""

    optimiser = _make_minimal_optimiser()
    save_path = str(tmp_path / "optimiser.json")
    save_to_json(optimiser, save_path)

    with open(save_path, 'r') as json_file:
        saved = json.load(json_file)

    saved = _downgrade_to_v1(saved)
    saved['optimiser']['settings']['allow_automatic_json_updates'] = False

    with open(save_path, 'w') as json_file:
        json.dump(saved, json_file)

    with pytest.raises(RuntimeError, match="schema version"):
        load_optimiser_from_state(save_path)


def test_load_v1_json_auto_migrates_with_allow_flag(tmp_path: Path) -> None:
    """Loading a v1 JSON with allow_automatic_json_updates=True must succeed and migrate the file."""

    optimiser = _make_minimal_optimiser()
    save_path = str(tmp_path / "optimiser.json")
    save_to_json(optimiser, save_path)

    with open(save_path, 'r') as json_file:
        saved = json.load(json_file)

    saved = _downgrade_to_v1(saved)
    saved['optimiser']['settings']['allow_automatic_json_updates'] = True

    with open(save_path, 'w') as json_file:
        json.dump(saved, json_file)

    loaded_optimiser = load_optimiser_from_state(save_path)
    assert loaded_optimiser is not None

    backup_path = save_path + '.bak'
    assert os.path.exists(backup_path), "Backup was not created during auto-migration."

    with open(save_path, 'r') as json_file:
        migrated = json.load(json_file)

    assert migrated.get('schema_version') == CURRENT_SCHEMA_VERSION


def test_migrate_json_is_no_op_on_current_version(tmp_path: Path) -> None:
    """Calling migrate_json on an already-current file must be a no-op (no backup created)."""

    optimiser = _make_minimal_optimiser()
    save_path = str(tmp_path / "optimiser.json")
    save_to_json(optimiser, save_path)

    migrate_json(save_path)

    backup_path = save_path + '.bak'
    assert not os.path.exists(backup_path), "Backup should not be created for an already-current file."
