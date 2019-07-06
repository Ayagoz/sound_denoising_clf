from SCD.pipeline import denosing_training
from pathlib import Path
import sys

data_path = sys.argv[1]
exp_path = sys.argv[2]

Path(exp_path).mkdir(exist_ok=True)

denosing_training(data_path, exp_path)