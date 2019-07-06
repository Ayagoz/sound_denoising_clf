import argparse
from SCD.pipeline import evaluate_on_test_all



if __name__ == '__main__':
    # *** Paths ***
    data_path = '/input/'

    #result path
    output_path = '/output/'

    #model path
    models_path = '/sound_denoising_clf/models/'

    # *** argparse ***
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', help='allows to calculate on CPU only', action='store_true')

    args = parser.parse_args()

    cuda = not args.cpu

    # *** Pipeline ***

    evaluate_on_test_all(path_data=data_path, path_model=models_path, exp_path=output_path, cuda=cuda)