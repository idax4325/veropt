import json
import shutil
from typing import Optional, Union

from veropt import bayesian_optimiser
from veropt.optimiser.objective import CallableObjective, InterfaceObjective
from veropt.optimiser.optimiser import BayesianOptimiser
from veropt.optimiser.optimiser_utility import OptimiserSettings
from veropt.optimiser.saver_loader_utility import SavableClass, TensorsAsListsEncoder
from veropt.optimiser.utility import get_arguments_of_function

# Loaded so it is known by load_optimiser_from_state
from veropt.interfaces.experiment_utility import ExperimentObjective  # noqa: F401

CURRENT_SCHEMA_VERSION = 4


def save_to_json(
        object_to_save: SavableClass,
        file_path: str,
) -> None:
    # TODO: prolly add some path stuff o:)

    save_dict = object_to_save.gather_dicts_to_save()
    save_dict['schema_version'] = CURRENT_SCHEMA_VERSION

    if '.json' in file_path:
        file_path_with_json = file_path
    else:
        file_path_with_json = file_path + '.json'

    with open(file_path_with_json, 'w') as json_file:
        json.dump(
            save_dict,
            json_file,
            cls=TensorsAsListsEncoder,
            indent=2
        )


def load_optimiser_from_state(
        file_name: str,
        allow_automatic_json_updates: Optional[bool] = None,
) -> 'BayesianOptimiser':
    """Load a ``BayesianOptimiser`` from a JSON file.

    Parameters
    ----------
    file_name:
        Path to the saved optimiser JSON.
    allow_automatic_json_updates:
        Override the ``allow_automatic_json_updates`` flag stored inside the JSON.
        Useful when the JSON predates the flag (or has it set to ``False``) but
        you want a one-off migration without editing the file manually.
        Passing ``True`` will migrate the file in-place and continue.
        Passing ``None`` (default) defers to whatever value is stored in the JSON.
    """

    with open(file_name, 'r') as json_file:
        saved_dict = json.load(json_file)

    schema_version = saved_dict.get('schema_version', 1)

    if schema_version < CURRENT_SCHEMA_VERSION:
        stored_flag = (
            saved_dict
            .get('optimiser', {})
            .get('settings', {})
            .get('allow_automatic_json_updates', False)
        )
        allow_updates = allow_automatic_json_updates if allow_automatic_json_updates is not None else stored_flag

        if allow_updates:
            migrate_json(file_name)
            with open(file_name, 'r') as json_file:
                saved_dict = json.load(json_file)
        else:
            raise RuntimeError(
                f"The optimiser JSON at '{file_name}' uses schema version {schema_version}, "
                f"but the current schema version is {CURRENT_SCHEMA_VERSION}. "
                f"To update the file, either:\n"
                f"  1. Run: from veropt.optimiser.optimiser_saver_loader import migrate_json\n"
                f"          migrate_json('{file_name}')\n"
                f"  2. Set 'allow_automatic_json_updates=True' in your optimiser config — future "
                f"saves will include this flag and auto-migrate on load."
            )

    return BayesianOptimiser.from_saved_state(saved_dict['optimiser'])


def _migrate_v1_to_v2(saved_dict: dict) -> dict:
    """Move noise fields (noise, noise_lower_bound, train_noise) from inside each kernel's
    'settings' dict to the top level of the kernel's state dict."""

    _NOISE_KEYS = frozenset({'noise', 'noise_lower_bound', 'train_noise'})

    try:
        model_dicts = (
            saved_dict
            ['optimiser']
            ['predictor']
            ['state']
            ['model']
            ['state']
            ['model_dicts']
        )
    except KeyError as missing_key:
        raise RuntimeError(
            f"Could not migrate JSON: unexpected structure. Missing key: {missing_key}. "
            "This file may already be in a non-standard format."
        ) from missing_key

    for model_key, model_dict in model_dicts.items():
        kernel_settings = model_dict['state'].get('settings', {})
        noise_values = {key: value for key, value in kernel_settings.items() if key in _NOISE_KEYS}
        clean_settings = {key: value for key, value in kernel_settings.items() if key not in _NOISE_KEYS}

        model_dicts[model_key]['state']['settings'] = clean_settings
        model_dicts[model_key]['state'].update(noise_values)

    saved_dict['schema_version'] = CURRENT_SCHEMA_VERSION

    return saved_dict


def migrate_json(file_path: str) -> None:
    """Migrate a saved optimiser JSON file to the current schema version.

    A backup is written to <file_path>.bak before the original is modified.
    The backup is written and flushed before touching the original, so both
    files are never in a corrupt state simultaneously.
    """

    file_path_with_json = file_path if '.json' in file_path else file_path + '.json'
    backup_path = file_path_with_json + '.bak'

    with open(file_path_with_json, 'r') as json_file:
        saved_dict = json.load(json_file)

    schema_version = saved_dict.get('schema_version', 1)

    if schema_version >= CURRENT_SCHEMA_VERSION:
        print(f"File '{file_path_with_json}' is already at schema version {schema_version}. No migration needed.")
        return

    # Write backup BEFORE modifying anything — use shutil.copy2 to preserve metadata
    shutil.copy2(file_path_with_json, backup_path)

    if schema_version < 2:
        saved_dict = _migrate_v1_to_v2(saved_dict)

    if schema_version < 3:
        saved_dict = _migrate_v2_to_v3(saved_dict)

    if schema_version < 4:
        saved_dict = _migrate_v3_to_v4(saved_dict)

    with open(file_path_with_json, 'w') as json_file:
        json.dump(saved_dict, json_file, cls=TensorsAsListsEncoder, indent=2)

    print(
        f"Migration complete: schema v{schema_version} → v{CURRENT_SCHEMA_VERSION}. "
        f"Backup saved at '{backup_path}'."
    )


def _migrate_v2_to_v3(saved_dict: dict) -> dict:
    """Fix train_inputs serialised as a nested list (from gpytorch tuple wrapping).

    In schema v2, gather_dicts_to_save saved model_with_data.train_inputs (a tuple),
    producing [[...data...]] in JSON — shape [1, n_points, n_vars] on reload.
    Schema v3 saves train_inputs[0] directly — shape [n_points, n_vars].
    This migration unwraps the extra leading dimension if present.
    """
    try:
        model_dicts = (
            saved_dict
            ['optimiser']
            ['predictor']
            ['state']
            ['model']
            ['state']
            ['model_dicts']
        )
    except KeyError as missing_key:
        raise RuntimeError(
            f"Could not migrate JSON: unexpected structure. Missing key: {missing_key}. "
            "This file may already be in a non-standard format."
        ) from missing_key

    for model_key, model_dict in model_dicts.items():
        train_inputs = model_dict['state'].get('train_inputs')
        if (
            train_inputs is not None
            and isinstance(train_inputs, list)
            and len(train_inputs) == 1
            and isinstance(train_inputs[0], list)
        ):
            model_dicts[model_key]['state']['train_inputs'] = train_inputs[0]

    saved_dict['schema_version'] = CURRENT_SCHEMA_VERSION

    return saved_dict


def _migrate_v3_to_v4(saved_dict: dict) -> dict:
    """Move train_noise from the model state to the objective state. Remove noise and noise_lower_bound
    from the model state — these are now derived at training time from objective.noise_std.

    train_noise is copied from the first single-model's state dict into the objective state.
    The objective gains noise_std_min and noise_std_max (both None, newly introduced fields)."""

    try:
        model_dicts = (
            saved_dict
            ['optimiser']
            ['predictor']
            ['state']
            ['model']
            ['state']
            ['model_dicts']
        )
        objective_state = saved_dict['optimiser']['objective']['state']
    except KeyError as missing_key:
        raise RuntimeError(
            f"Could not migrate JSON: unexpected structure. Missing key: {missing_key}. "
            "This file may already be in a non-standard format."
        ) from missing_key

    # Extract train_noise from the first model (all models share the same value)
    first_model_state = next(iter(model_dicts.values()))['state']
    train_noise = first_model_state.get('train_noise', False)

    # Remove the now-redundant noise/noise_lower_bound fields from every model state
    for model_key in model_dicts:
        model_dicts[model_key]['state'].pop('noise', None)
        model_dicts[model_key]['state'].pop('noise_lower_bound', None)

    # Add the new noise fields to the objective
    objective_state['train_noise'] = train_noise
    objective_state.setdefault('noise_std_min', None)
    objective_state.setdefault('noise_std_max', None)

    saved_dict['schema_version'] = CURRENT_SCHEMA_VERSION

    return saved_dict


def load_optimiser_from_settings(
        file_name: str,
        objective: Union[InterfaceObjective, CallableObjective],
) -> 'BayesianOptimiser':

    with open(file_name, 'r') as json_file:
        settings_dict = json.load(json_file)

    required_arguments = get_arguments_of_function(
        function=bayesian_optimiser,
        argument_type='required',
        excluded_arguments=['objective']
    )

    for required_parameter in required_arguments:
        assert required_parameter in settings_dict, (
            f"The top level of an optimiser settings file must contain (at least) {required_arguments} "
            f"but got {list(settings_dict.keys())}"
        )

    all_arguments = get_arguments_of_function(
        function=bayesian_optimiser,
        excluded_arguments=['objective', 'kwargs']
    )

    all_arguments += get_arguments_of_function(
        function=OptimiserSettings.__init__,
        excluded_arguments=['self', 'n_objectives'] + all_arguments
    )

    for key in settings_dict.keys():
        assert key in all_arguments, f"Key '{key}' not recognised. Must be one of {all_arguments}."

    return bayesian_optimiser(
        objective=objective,
        **settings_dict
    )
