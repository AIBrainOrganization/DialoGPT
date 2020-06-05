python train_reinforce.py --train_input_file data/reinforce/train.200len.db --eval_input_file data/reinforce/valid.200len.db --output_dir ./models/reinforce --seed 42 --max_seq_length 200 --train_batch_size 512 --gradient_accumulation_steps 8 --eval_batch_size 64 --learning_rate 1e-5 --num_optim_steps 43390 --valid_step 1085 --warmup_steps 4000 --normalize_data true --fp16 true --lr_schedule noam --loss_scale 0.0 --no_token_id true --pbar true
# --init_checkpoint /home/calee/git/DialoGPT/models/dcinside-news_new/GPT2.1e-05.64.2gpu.2020-05-14110855/GP2-pretrain-step-30960.pkl --continue_from 30962
