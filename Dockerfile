FROM tensorflow/tensorflow:latest-gpu

COPY ./requirements.txt .

COPY ./run_main.sh .
RUN chmod +x ./run_main.sh

RUN export GIT_PYTHON_REFRESH=quiet

RUN pip install -r ./requirements.txt

