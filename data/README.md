## How to create a DB
1. Prepare a tsv file.  
It has two columns.  
First column is source sentences and the second column is a target sentence.  
Source sentences are separated by EOS.  
Each sentence has 0.0 or 1.0 before the sentence, which is a weight to indicate whether the sentence should be trained or not. (1.0 means the sentence needs to be learned.)

2. Run prepro.py
```bash
python prepro.py --corpus {path to the tsv} --max_seq_len 200
```
