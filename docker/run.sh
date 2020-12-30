#!/bin/bash
docker run --gpus 1 -v "$PWD/..":/git/DialoGPT -p 5000:5000 calee/dialogpt /bin/bash --login -c "conda activate LSP; python server.py"
