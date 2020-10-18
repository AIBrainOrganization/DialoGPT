FROM nvidia/cuda:10.1-base
WORKDIR /git/DialoGPT
RUN apt update
RUN apt install -y wget
RUN apt install -y gcc
RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
RUN bash Anaconda3-2020.02-Linux-x86_64.sh -b -p ~/anaconda3
SHELL ["/bin/bash", "--login", "-c"]
RUN sed -i '1 i\. ~/anaconda3/etc/profile.d/conda.sh' ~/.bashrc
RUN echo ~/.bashrc
COPY DialoGPT/LSP-linux.yml /root/LSP-linux.yml
COPY KoGPT2/requirements.txt /root/requirements.txt
RUN conda env create -f /root/LSP-linux.yml -n LSP
RUN conda run -n LSP pip install -r /root/requirements.txt
RUN conda run -n LSP pip install flask
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV FLASK_APP=server.py
