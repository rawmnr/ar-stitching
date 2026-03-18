import numpy as np
from pathlib import Path
from stitching.contracts import ScenarioConfig
from stitching.harness.evaluator import load_candidate_module, evaluate_candidate_on_scenario

scenario_path = Path('scenarios/s17_highres_circular.yaml')
config = ScenarioConfig.from_yaml(scenario_path)

for name, path in [('GLS', 'src/stitching/editable/gls/baseline.py'), ('SCS', 'src/stitching/editable/scs/baseline.py'), ('SIAC', 'src/stitching/editable/siac/baseline.py'), ('SIAC+Reg', 'src/stitching/editable/siac_reg/baseline.py')]:
    candidate = load_candidate_module(Path(path))
    report = evaluate_candidate_on_scenario(candidate, scenario_path)
    print(f'{name}: {report.signal_metrics.get("rms_detrended"):.6f}')
