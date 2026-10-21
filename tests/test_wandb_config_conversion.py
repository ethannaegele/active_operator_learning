import tempfile
from pathlib import Path

import yaml

from al4pde.utils import convert_wandb_config


def test_convert_wandb_config_unwraps_settings_and_removes_metadata():
    raw_config = {
        "_wandb": {"value": {"email": "person@example.com", "host": "cluster"}},
        "task": {"value": {"data_gen": {"batch_size": 4}}},
        "acquisition": {"value": {"batch_size": "${task.data_gen.batch_size}"}},
        "wandb": {"value": {"name": "example_${seed}"}},
        "seed": {"value": 7},
    }

    with tempfile.TemporaryDirectory() as directory:
        config_path = Path(directory) / "export.yaml"
        with config_path.open("w") as file:
            yaml.safe_dump(raw_config, file)

        convert_wandb_config(config_path)

        with config_path.open() as file:
            converted = yaml.safe_load(file)

    assert "_wandb" not in converted
    assert converted["task"]["data_gen"]["batch_size"] == 4
    assert converted["acquisition"]["batch_size"] == "${task.data_gen.batch_size}"
    assert converted["wandb"]["name"] == "example_${seed}"
    assert converted["seed"] == 7
    assert converted["hydra"]["run"]["dir"] == "."
