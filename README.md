## Requirements
You may be able to use the library without this requirements, but the library is tested only from the below environment.
- One or more Titan V graphics cards (driver should be installed)
- Ubuntu 18.04.4 LTS

## Installation
1. Install Anaconda  
```bash
cd ~
mkdir Downloads
cd ~/Downloads
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh
```
Follow instructions.  
Answer 'yes' to prepend the Anaconda3 install location to PATH in your .bashrc.  
Answer 'no' not to install Microsoft VSCode.


2. Install NVIDIA CUDA Toolkit  


3. Clone DialoGPT  
```bash
git clone http://EmoCA.kaist.ac.kr/calee/DialoGPT.git
```


4. Create and activate conda environment  
```bash
cd DialoGPT
conda create -f LSP-linux.yml -n LSP
conda activate LSP
```

5. Clone and install KoGPT2  
```bash
git clone http://EmoCA.kaist.ac.kr/calee/KoGPT2.git
cd KoGPT2
pip install -r requirements.txt
pip install .
```


6. Copy files  
data/train.200len.db
data/valid.200len.db

## Training
```bash
bash run.sh
```
