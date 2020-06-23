FROM nvidia/cuda:10.2-devel
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
COPY KoGPT2 /git/KoGPT2
RUN conda run -n LSP pip install /git/KoGPT2
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV FLASK_APP=server.py
WORKDIR /home/calee/git/DialoGPT
SHELL ["/bin/bash", "-c"]
RUN mkdir -p /home/calee/git
RUN apt install -y git
RUN cd /home/calee/git/ && git clone https://github.com/NVIDIA/apex
RUN cd /home/calee/git/apex && git reset --hard 3d01e4a0a188cc8df54bc6e44cf5eb40ff6b4cc5
SHELL ["/bin/bash", "--login", "-c"]
RUN cd /home/calee/git/apex && conda run -n LSP pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
