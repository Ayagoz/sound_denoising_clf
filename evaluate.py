from SCD.pipeline import evaluate_on_test_all
import sys
data_path = sys.argv[1]
model_path = sys.argv[2]
exp_path = sys.argv[3]

evaluate_on_test_all(path_data= data_path, path_model=model_path, exp_path=exp_path, cuda=False)