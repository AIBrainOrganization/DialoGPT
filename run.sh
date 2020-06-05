python LSP_train.py --train_input_file /home/calee/data/dcinside/news_new/tsv_formatted/train.200len.reverse.db --eval_input_file /home/calee/data/dcinside/news_new/tsv_formatted/valid.200len.reverse.db --output_dir ./models/output_model --seed 42 --max_seq_length 200 --train_batch_size 512 --gradient_accumulation_steps 8 --eval_batch_size 64 --learning_rate 1e-5 --num_optim_steps 33135 --valid_step 1657 --warmup_steps 4000 --normalize_data true --fp16 true --lr_schedule noam --loss_scale 0.0 --no_token_id true --pbar true
# --init_checkpoint /home/calee/git/DialoGPT/models/output_model/GPT2.1e-05.64.2gpu.2020-03-20190608/GP2-pretrain-step-10880.pkl --continue_from 10881
