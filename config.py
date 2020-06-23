# device for forward and backward model
device_f = 0
device_r = 0

# sampling parameters
top_k = 0

num_samples = 20
ALPHA = 0.5  # 1에 가까우면 기본 모델에 집중하고 0에 가까우면 reverse 모델에 집중합니다.
BETA = 0.5  # 전체에서 Q function 비중
top_p = 1.0

min_p_alpha = 2

# default paths
vocab_path = 'models/kogpt2_news_wiki_ko_cased_818bfa919d.spiece'
model_path = 'models/output_model/' \
    'GPT2.1e-05.64.2gpu.2020-03-24160510/GP2-pretrain-step-20672.pkl'
reverse_model_path = 'models/output_model/' \
    'GPT2.1e-05.64.2gpu.2020-03-26162233/GP2-pretrain-step-8704.pkl'
# model_path = 'models/dcinside-news_new/' \
#     'GPT2.1e-05.64.2gpu.2020-05-19161142/GP2-pretrain-step-61920.pkl'
# reverse_model_path = 'models/dcinside-news_new/' \
#     'GPT2.1e-05.64.2gpu.2020-05-20084102/GP2-pretrain-step-61920.pkl'
