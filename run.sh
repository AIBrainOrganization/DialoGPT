python LSP_train.py --train_input_file data/train.200len.db --eval_input_file data/valid.200len.db --output_dir models/dcinside/10/fine --seed 42 --max_seq_length 200 --train_batch_size 512 --gradient_accumulation_steps 8 --eval_batch_size 64 --learning_rate 1e-5 --num_optim_steps 43523 --valid_step 1088 --warmup_steps 1088 --normalize_data true --fp16 true --lr_schedule noam --loss_scale 0.0 --no_token_id true --pbar true --init_checkpoint models/dcinside/10/GPT2.1e-05.64.2gpu.2020-10-26133346/GP2-pretrain-step-310400.pkl
# num_optim_steps = `wc -l train.tsv` / train_batch_size * 20 epochs
# valid_step = `wc -l train.tsv` / train_batch_size * 0.5 epoch
# warmup_steps = valid_step