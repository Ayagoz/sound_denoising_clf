# sound classification and denosing

1. Clone repository:

`git clone https://github.com/Ayagoz/sound_denoising_clf.git`

2. Build docker:

`nvidia-docker build -t scd -f ndl.dockerfile .`

`nvidia-docker run -dit -v [TEST-PRE]:/input/pre:ro -v /output scd`

Output in the comand line will be > [CONTAINER-ID]

3. Docker execution:

    `nvidia-docker exec [CONTAINER-ID] python /sound_denoising_clf/entrypoint.py`

    `nvidia-docker cp [CONTAINER-ID]:/output [RESULT-LOCATION]`

4. Docker deletion:

`nvidia-docker stop [CONTAINER-ID]`

`nvidia-docker rm -v [CONTAINER-ID]`