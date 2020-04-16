# device for forward and backward model
device_f = 'cuda'
device_r = 'cpu'

# sampling parameters
top_k = 50

num_samples = 40
ALPHA = 0.5  # 1에 가까우면 기본 모델에 집중하고 0에 가까우면 reverse 모델에 집중합니다.
top_p = 0.9
