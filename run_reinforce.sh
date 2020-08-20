python -u train_reinforce.py --train_input_file data/reinforce/acryl_emotion/train.200len.2turn.db --eval_input_file data/reinforce/acryl_emotion/valid.200len.2turn.db --output_dir ./models/reinforce/acryl_emotion --seed 42 --max_seq_length 200 --train_batch_size 512 --gradient_accumulation_steps 8 --eval_batch_size 64 --learning_rate 1e-5 --num_optim_steps 5735 --valid_step 143 --warmup_steps 143 --normalize_data true --fp16 false --lr_schedule noam --loss_scale 0.0 --no_token_id true --pbar true --decrease_exponent 0.1
# --init_checkpoint /home/calee/git/DialoGPT/models/reinforce/acryl_no_neutral/GPT2.1e-05.64.2gpu.2020-07-22163612/GP2-pretrain-step-560.pkl
# --continue_from 30962
# num_optim_steps = `wc -l train.tsv` / train_batch_size * 20 epochs
# valid_step = `wc -l train.tsv` / train_batch_size * 0.5 epoch
# warmup_steps = valid_step
# acryl_emotion: 5735, 143
# acryl_no_neutral: 643, 16
