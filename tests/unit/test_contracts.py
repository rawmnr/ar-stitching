from pathlib import Path

from stitching.contracts import ScenarioConfig


def test_scenario_config_loads_identity_yaml() -> None:
    config = ScenarioConfig.from_yaml(Path("scenarios/s00_identity.yaml"))

    assert config.scenario_id == "s00_identity"
    assert config.grid_shape == (8, 8)
    assert config.scan_offsets == ((0.0, 0.0),)
