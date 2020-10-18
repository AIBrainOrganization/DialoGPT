# device for forward and backward model
device_f = 'cuda'
device_r = 'cpu'

# sampling parameters
top_k = 50

num_samples = 40
ALPHA = 0.5  # 1에 가까우면 기본 모델에 집중하고 0에 가까우면 reverse 모델에 집중합니다.
top_p = 0.9

# default paths
vocab_path = 'models/kogpt2_news_wiki_ko_cased_818bfa919d.spiece'
# model_path = 'models/output_model/' \
#     'GPT2.1e-05.64.2gpu.2020-03-24160510/GP2-pretrain-step-20672.pkl'
model_path='/home/calee/git/DialoGPT/models/kaist/GP2-pretrain-step-8041.pkl'
reverse_model_path = 'models/output_model/' \
    'GPT2.1e-05.64.2gpu.2020-03-26162233/GP2-pretrain-step-8704.pkl'
