python LSP_train.py --train_input_file data/dcinside/tsv/10/train.200len.db --eval_input_file data/dcinside/tsv/10/valid.200len.db --output_dir ./models/dcinside/10 --seed 42 --max_seq_length 200 --train_batch_size 512 --gradient_accumulation_steps 8 --eval_batch_size 64 --learning_rate 1e-5 --num_optim_steps 310400 --valid_step 7760 --warmup_steps 7760 --normalize_data true --fp16 true --lr_schedule noam --loss_scale 0.0 --no_token_id true --pbar true
# --init_checkpoint /home/calee/git/DialoGPT/models/output_model/GPT2.1e-05.64.2gpu.2020-03-24160510/GP2-pretrain-step-20672.pkl
# num_optim_steps = `wc -l train.tsv` / train_batch_size * 20 epochs
# valid_step = `wc -l train.tsv` / train_batch_size * 0.5 epoch
# warmup_steps = valid_step