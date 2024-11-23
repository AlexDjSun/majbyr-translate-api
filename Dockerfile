FROM nvidia/cuda:12.6.2-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \ 
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    curl \
    libsndfile1 \
    build-essential \
    g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

COPY . /

RUN pip install --no-cache-dir -r requirements.txt

RUN [ "python3", "-c", "import nltk; nltk.download('punkt', download_dir='/usr/local/nltk_data')" ]

ENV NLTK_DATA=/usr/local/nltk_data

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]