# sound classification and denosing
This repository contents a basic models to classify if sound is noisy and denoise it.
As input data the mel-spectrogram is used. 
The data folders should be organized as follows:

* [DATA FOLDER]
    * `train`
        * `clean`
        * `noisy`
    * `val`
        * `clean`
        * `noisy`
    * `test`
        * `clean`
        * `noisy`

1. Clone repository:

    `git clone https://github.com/Ayagoz/sound_denoising_clf.git`

2. Build docker:

    `nvidia-docker build -t scd -f ndl.dockerfile .`

    `nvidia-docker run -dit -v [TEST]:/input/pre:ro -v /output scd`

    Output in the comand line will be > [CONTAINER-ID]

    Docker execution:

    `nvidia-docker exec [CONTAINER-ID] python /sound_denoising_clf/entrypoint.py`

    `nvidia-docker cp [CONTAINER-ID]:/output [RESULT-LOCATION]`

    Docker deletion:

    `nvidia-docker stop [CONTAINER-ID]`

    `nvidia-docker rm -v [CONTAINER-ID]`

3. Another way:
    From console:

    `python SCD/evaluate.py [DATA PATH] [MODEL WEIGHTS PATH] [OUTPUT PATH]`