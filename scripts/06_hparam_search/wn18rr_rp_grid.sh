#!/bin/bash



# start=1
# end=4

# run from the repository root with the PathE environment activated and pathe installed (pip install -e .)

# python -m pathe.runner train pathe \
#   --expname llf:ce_ea:recurrent_nal:1_nah:1_ch:1_bs:4096_d:0.2_mp:8_nn:0_n:4_nel:1_df:128_ed:64_ag:1_ls:0.01_cw_1291 \
#   --wandb_project wn18rr-rp \
#   --log_dir artifacts/runs/relation_prediction/wn18rr-rp \
#   --train_paths artifacts/data/path_datasets/wn18rr/train/ \
#   --valid_paths artifacts/data/path_datasets/wn18rr/val/ \
#   --test_paths artifacts/data/path_datasets/wn18rr/test/ \
#   --path_setup 20_20 --max_ppt 8 --batch_size 1024 \
#   --num_negatives 0 --val_num_negatives 0 \
#   --nhead 4 --num_encoder_layers 1 \
#   --ent_aggregation recurrent --num_agg_layers 1 \
#   --num_agg_heads 1 --context_heads 1 \
#   --dim_feedforward 128 --embedding_dim 64 \
#   --lp_loss_fn ce --accumulate_gradient 4 \
#   --monitor valid_loss --patience 10 --dropout 0.2 \
#   --lrate 0.001 --weight_decay 0 --label_smoothing 0.01 \
#   --device cuda --num_devices 1 --num_workers 0 --seed 46

# python -m pathe.runner train pathe \
#   --expname llf:ce_ea:recurrent_nal:1_nah:1_ch:0_bs:4096_mp:4_nn:0_n:4_nel:1_df:128_ed:64_ag:1_ls:0.01_1035 \
#   --wandb_project wn18rr-rp \
#   --log_dir artifacts/runs/relation_prediction/wn18rr-rp \
#   --train_paths artifacts/data/path_datasets/wn18rr/train/ \
#   --valid_paths artifacts/data/path_datasets/wn18rr/val/ \
#   --test_paths artifacts/data/path_datasets/wn18rr/test/ \
#   --path_setup 20_20 --max_ppt 4 --batch_size 1024 \
#   --num_negatives 0 --val_num_negatives 0 \
#   --nhead 4 --num_encoder_layers 1 \
#   --ent_aggregation recurrent --num_agg_layers 1 \
#   --num_agg_heads 1 --context_heads 0 \
#   --dim_feedforward 128 --embedding_dim 64 \
#   --lp_loss_fn ce --accumulate_gradient 4 \
#   --monitor valid_loss --patience 10 --dropout 0.1 \
#   --lrate 0.001 --weight_decay 0 --label_smoothing 0.01 \
#   --device cuda --num_devices 1 --num_workers 0 --seed 46

# python -m pathe.runner train pathe \
#   --expname llf:ce_ea:recurrent_nal:1_nah:1_ch:2_bs:4096_d:0.2_mp:8_nn:0_nel:1_df:128_ed:32_ag:1_ls:0.1_1520 \
#   --wandb_project wn18rr-rp \
#   --log_dir artifacts/runs/relation_prediction/wn18rr-rp \
#   --train_paths artifacts/data/path_datasets/wn18rr/train/ \
#   --valid_paths artifacts/data/path_datasets/wn18rr/val/ \
#   --test_paths artifacts/data/path_datasets/wn18rr/test/ \
#   --path_setup 20_20 --max_ppt 8 --batch_size 1024 \
#   --num_negatives 0 --val_num_negatives 0 \
#   --nhead 2 --num_encoder_layers 1 \
#   --ent_aggregation recurrent --num_agg_layers 1 \
#   --num_agg_heads 1 --context_heads 2 \
#   --dim_feedforward 128 --embedding_dim 32 \
#   --lp_loss_fn ce --accumulate_gradient 4 \
#   --monitor valid_loss --patience 10 --dropout 0.2 \
#   --lrate 0.001 --weight_decay 0 --label_smoothing 0.1 \
#   --device cuda --num_devices 1 --num_workers 0 --seed 46

# python -m pathe.runner train pathe \
#   --expname llf:ce_ea:avg_nal:1_nah:1_ch:0_bs:4096_mp:8_nn:0_n:4_nel:2_df:128_ed:64_ag:1_ls:0.1_cw_46 \
#   --wandb_project wn18rr-rp \
#   --log_dir artifacts/runs/relation_prediction/wn18rr-rp \
#   --train_paths artifacts/data/path_datasets/wn18rr/train/ \
#   --valid_paths artifacts/data/path_datasets/wn18rr/val/ \
#   --test_paths artifacts/data/path_datasets/wn18rr/test/ \
#   --path_setup 20_20 --max_ppt 8 --batch_size 1024 \
#   --num_negatives 0 --val_num_negatives 0 \
#   --nhead 4 --num_encoder_layers 2 \
#   --ent_aggregation avg --num_agg_layers 1 \
#   --num_agg_heads 1 --context_heads 0 \
#   --dim_feedforward 128 --embedding_dim 64 \
#   --lp_loss_fn ce --accumulate_gradient 4 \
#   --monitor valid_loss --patience 10 --dropout 0.1 \
#   --lrate 0.001 --weight_decay 0 --label_smoothing 0.1 \
#   --device cuda --num_devices 1 --num_workers 0 --seed 46

# python -m pathe.runner train pathe \
#   --expname llf:ce_ea:transformer_nal:1_nah:2_ch:0_bs:4096_mp:2_nn:0_n:4_nel:2_df:128_ed:32_ag:1_ls:0.1_cw_396 \
#   --wandb_project wn18rr-rp \
#   --log_dir artifacts/runs/relation_prediction/wn18rr-rp \
#   --train_paths artifacts/data/path_datasets/wn18rr/train/ \
#   --valid_paths artifacts/data/path_datasets/wn18rr/val/ \
#   --test_paths artifacts/data/path_datasets/wn18rr/test/ \
#   --path_setup 20_20 --max_ppt 2 --batch_size 1024 \
#   --num_negatives 0 --val_num_negatives 0 \
#   --nhead 4 --num_encoder_layers 2 \
#   --ent_aggregation transformer --num_agg_layers 1 \
#   --num_agg_heads 2 --context_heads 0 \
#   --dim_feedforward 128 --embedding_dim 32 \
#   --lp_loss_fn ce --accumulate_gradient 4 \
#   --monitor valid_loss --patience 10 --dropout 0.1 \
#   --lrate 0.001 --weight_decay 0 --label_smoothing 0.1 \
#   --device cuda --num_devices 1 --num_workers 0 --seed 46

# python -m pathe.runner train pathe \
#   --expname llf:ce_ea:recurrent_nal:1_nah:1_ch:1_bs:4096_mp:4_nn:0_nel:2_df:128_ed:64_ag:1_ls:0.01_1223 \
#   --wandb_project wn18rr-rp \
#   --log_dir artifacts/runs/relation_prediction/wn18rr-rp \
#   --train_paths artifacts/data/path_datasets/wn18rr/train/ \
#   --valid_paths artifacts/data/path_datasets/wn18rr/val/ \
#   --test_paths artifacts/data/path_datasets/wn18rr/test/ \
#   --path_setup 20_20 --max_ppt 4 --batch_size 1024 \
#   --num_negatives 0 --val_num_negatives 0 \
#   --nhead 2 --num_encoder_layers 2 \
#   --ent_aggregation recurrent --num_agg_layers 1 \
#   --num_agg_heads 1 --context_heads 1 \
#   --dim_feedforward 128 --embedding_dim 64 \
#   --lp_loss_fn ce --accumulate_gradient 4 \
#   --monitor valid_loss --patience 10 --dropout 0.1 \
#   --lrate 0.001 --weight_decay 0 --label_smoothing 0.01 \
#   --device cuda --num_devices 1 --num_workers 0 --seed 46

# python -m pathe.runner train pathe \
#   --expname llf:ce_ea:transformer_nal:2_nah:1_ch:0_bs:4096_d:0.2_mp:4_nn:0_n:4_nel:1_df:128_ed:32_ag:1_ls:0.1_cw_696 \
#   --wandb_project wn18rr-rp \
#   --log_dir artifacts/runs/relation_prediction/wn18rr-rp \
#   --train_paths artifacts/data/path_datasets/wn18rr/train/ \
#   --valid_paths artifacts/data/path_datasets/wn18rr/val/ \
#   --test_paths artifacts/data/path_datasets/wn18rr/test/ \
#   --path_setup 20_20 --max_ppt 4 --batch_size 1024 \
#   --num_negatives 0 --val_num_negatives 0 \
#   --nhead 4 --num_encoder_layers 1 \
#   --ent_aggregation transformer --num_agg_layers 2 \
#   --num_agg_heads 1 --context_heads 0 \
#   --dim_feedforward 128 --embedding_dim 32 \
#   --lp_loss_fn ce --accumulate_gradient 4 \
#   --monitor valid_loss --patience 10 --dropout 0.2 \
#   --lrate 0.001 --weight_decay 0 --label_smoothing 0.1 \
#   --device cuda --num_devices 1 --num_workers 0 --seed 46

# python -m pathe.runner train pathe \
#   --expname llf:ce_ea:transformer_nal:2_nah:1_ch:0_bs:4096_mp:4_nn:0_n:4_nel:2_df:128_ed:64_ag:1_ls:0.1_cw_606 \
#   --wandb_project wn18rr-rp \
#   --log_dir artifacts/runs/relation_prediction/wn18rr-rp \
#   --train_paths artifacts/data/path_datasets/wn18rr/train/ \
#   --valid_paths artifacts/data/path_datasets/wn18rr/val/ \
#   --test_paths artifacts/data/path_datasets/wn18rr/test/ \
#   --path_setup 20_20 --max_ppt 4 --batch_size 1024 \
#   --num_negatives 0 --val_num_negatives 0 \
#   --nhead 4 --num_encoder_layers 2 \
#   --ent_aggregation transformer --num_agg_layers 2 \
#   --num_agg_heads 1 --context_heads 0 \
#   --dim_feedforward 128 --embedding_dim 64 \
#   --lp_loss_fn ce --accumulate_gradient 4 \
#   --monitor valid_loss --patience 10 --dropout 0.1 \
#   --lrate 0.001 --weight_decay 0 --label_smoothing 0.1 \
#   --device cuda --num_devices 1 --num_workers 0 --seed 46

# python -m pathe.runner train pathe \
#   --expname llf:ce_ea:recurrent_nal:1_nah:1_ch:0_bs:4096_d:0.2_mp:8_nn:0_n:4_nel:2_df:128_ed:64_ag:1_ls:0.01_1151 \
#   --wandb_project wn18rr-rp \
#   --log_dir artifacts/runs/relation_prediction/wn18rr-rp \
#   --train_paths artifacts/data/path_datasets/wn18rr/train/ \
#   --valid_paths artifacts/data/path_datasets/wn18rr/val/ \
#   --test_paths artifacts/data/path_datasets/wn18rr/test/ \
#   --path_setup 20_20 --max_ppt 8 --batch_size 1024 \
#   --num_negatives 0 --val_num_negatives 0 \
#   --nhead 4 --num_encoder_layers 2 \
#   --ent_aggregation recurrent --num_agg_layers 1 \
#   --num_agg_heads 1 --context_heads 0 \
#   --dim_feedforward 128 --embedding_dim 64 \
#   --lp_loss_fn ce --accumulate_gradient 4 \
#   --monitor valid_loss --patience 10 --dropout 0.2 \
#   --lrate 0.001 --weight_decay 0 --label_smoothing 0.01 \
#   --device cuda --num_devices 1 --num_workers 0 --seed 46

# python -m pathe.runner train pathe \
#   --expname llf:ce_ea:transformer_nal:2_nah:2_ch:0_bs:4096_mp:8_nn:0_n:4_nel:1_df:128_ed:32_ag:1_ls:0.1_856 \
#   --wandb_project wn18rr-rp \
#   --log_dir artifacts/runs/relation_prediction/wn18rr-rp \
#   --train_paths artifacts/data/path_datasets/wn18rr/train/ \
#   --valid_paths artifacts/data/path_datasets/wn18rr/val/ \
#   --test_paths artifacts/data/path_datasets/wn18rr/test/ \
#   --path_setup 20_20 --max_ppt 8 --batch_size 1024 \
#   --num_negatives 0 --val_num_negatives 0 \
#   --nhead 4 --num_encoder_layers 1 \
#   --ent_aggregation transformer --num_agg_layers 2 \
#   --num_agg_heads 2 --context_heads 0 \
#   --dim_feedforward 128 --embedding_dim 32 \
#   --lp_loss_fn ce --accumulate_gradient 4 \
#   --monitor valid_loss --patience 10 --dropout 0.1 \
#   --lrate 0.001 --weight_decay 0 --label_smoothing 0.1 \
#   --device cuda --num_devices 1 --num_workers 0 --seed 46

# python -m pathe.runner train pathe \
#   --expname llf:ce_ea:avg_nal:1_nah:1_ch:0_bs:4096_d:0.2_mp:4_nn:0_nel:1_df:128_ed:64_ag:1_ls:0.01_163 \
#   --wandb_project wn18rr-rp \
#   --log_dir artifacts/runs/relation_prediction/wn18rr-rp \
#   --train_paths artifacts/data/path_datasets/wn18rr/train/ \
#   --valid_paths artifacts/data/path_datasets/wn18rr/val/ \
#   --test_paths artifacts/data/path_datasets/wn18rr/test/ \
#   --path_setup 20_20 --max_ppt 4 --batch_size 1024 \
#   --num_negatives 0 --val_num_negatives 0 \
#   --nhead 2 --num_encoder_layers 1 \
#   --ent_aggregation avg --num_agg_layers 1 \
#   --num_agg_heads 1 --context_heads 0 \
#   --dim_feedforward 128 --embedding_dim 64 \
#   --lp_loss_fn ce --accumulate_gradient 4 \
#   --monitor valid_loss --patience 10 --dropout 0.2 \
#   --lrate 0.001 --weight_decay 0 --label_smoothing 0.01 \
#   --device cuda --num_devices 1 --num_workers 0 --seed 46

# python -m pathe.runner train pathe \
#   --expname llf:ce_ea:recurrent_nal:1_nah:1_ch:2_bs:4096_d:0.2_mp:8_nn:0_nel:2_df:128_ed:64_ag:1_ls:0.1_cw_1478 \
#   --wandb_project wn18rr-rp \
#   --log_dir artifacts/runs/relation_prediction/wn18rr-rp \
#   --train_paths artifacts/data/path_datasets/wn18rr/train/ \
#   --valid_paths artifacts/data/path_datasets/wn18rr/val/ \
#   --test_paths artifacts/data/path_datasets/wn18rr/test/ \
#   --path_setup 20_20 --max_ppt 8 --batch_size 1024 \
#   --num_negatives 0 --val_num_negatives 0 \
#   --nhead 2 --num_encoder_layers 2 \
#   --ent_aggregation recurrent --num_agg_layers 1 \
#   --num_agg_heads 1 --context_heads 2 \
#   --dim_feedforward 128 --embedding_dim 64 \
#   --lp_loss_fn ce --accumulate_gradient 4 \
#   --monitor valid_loss --patience 10 --dropout 0.2 \
#   --lrate 0.001 --weight_decay 0 --label_smoothing 0.1 \
#   --device cuda --num_devices 1 --num_workers 0 --seed 46

# python -m pathe.runner train pathe \
#   --expname llf:ce_ea:transformer_nal:2_nah:1_ch:0_bs:4096_d:0.2_mp:8_nn:0_nel:1_df:128_ed:32_ag:1_ls:0.01_753 \
#   --wandb_project wn18rr-rp \
#   --log_dir artifacts/runs/relation_prediction/wn18rr-rp \
#   --train_paths artifacts/data/path_datasets/wn18rr/train/ \
#   --valid_paths artifacts/data/path_datasets/wn18rr/val/ \
#   --test_paths artifacts/data/path_datasets/wn18rr/test/ \
#   --path_setup 20_20 --max_ppt 8 --batch_size 1024 \
#   --num_negatives 0 --val_num_negatives 0 \
#   --nhead 2 --num_encoder_layers 1 \
#   --ent_aggregation transformer --num_agg_layers 2 \
#   --num_agg_heads 1 --context_heads 0 \
#   --dim_feedforward 128 --embedding_dim 32 \
#   --lp_loss_fn ce --accumulate_gradient 4 \
#   --monitor valid_loss --patience 10 --dropout 0.2 \
#   --lrate 0.001 --weight_decay 0 --label_smoothing 0.01 \
#   --device cuda --num_devices 1 --num_workers 0 --seed 46

# python -m pathe.runner train pathe \
#   --expname llf:ce_ea:transformer_nal:2_nah:1_ch:0_bs:4096_mp:8_nn:0_n:4_nel:1_df:128_ed:64_ag:1_ls:0.01_667 \
#   --wandb_project wn18rr-rp \
#   --log_dir artifacts/runs/relation_prediction/wn18rr-rp \
#   --train_paths artifacts/data/path_datasets/wn18rr/train/ \
#   --valid_paths artifacts/data/path_datasets/wn18rr/val/ \
#   --test_paths artifacts/data/path_datasets/wn18rr/test/ \
#   --path_setup 20_20 --max_ppt 8 --batch_size 1024 \
#   --num_negatives 0 --val_num_negatives 0 \
#   --nhead 4 --num_encoder_layers 1 \
#   --ent_aggregation transformer --num_agg_layers 2 \
#   --num_agg_heads 1 --context_heads 0 \
#   --dim_feedforward 128 --embedding_dim 64 \
#   --lp_loss_fn ce --accumulate_gradient 4 \
#   --monitor valid_loss --patience 10 --dropout 0.1 \
#   --lrate 0.001 --weight_decay 0 --label_smoothing 0.01 \
#   --device cuda --num_devices 1 --num_workers 0 --seed 46

python -m pathe.runner train pathe \
  --expname llf:ce_ea:recurrent_nal:2_nah:1_ch:1_bs:4096_mp:8_nn:0_nel:2_df:128_ed:64_ag:1_ls:0.1_cw_1766 \
  --wandb_project wn18rr-rp \
  --log_dir artifacts/runs/relation_prediction/wn18rr-rp \
  --train_paths artifacts/data/path_datasets/wn18rr/train/ \
  --valid_paths artifacts/data/path_datasets/wn18rr/val/ \
  --test_paths artifacts/data/path_datasets/wn18rr/test/ \
  --path_setup 20_20 --max_ppt 8 --batch_size 512 \
  --num_negatives 0 --val_num_negatives 0 \
  --nhead 2 --num_encoder_layers 2 \
  --ent_aggregation recurrent --num_agg_layers 2 \
  --num_agg_heads 1 --context_heads 1 \
  --dim_feedforward 128 --embedding_dim 64 \
  --lp_loss_fn ce --accumulate_gradient 8 \
  --monitor valid_loss --patience 10 --dropout 0.1 \
  --lrate 0.001 --weight_decay 0 --label_smoothing 0.1 \
  --device cuda --num_devices 1 --num_workers 0 --seed 46

python -m pathe.runner train pathe \
  --expname llf:ce_ea:recurrent_nal:1_nah:1_ch:0_bs:4096_mp:4_nn:0_nel:2_df:128_ed:64_ag:1_ls:0.1_cw_982 \
  --wandb_project wn18rr-rp \
  --log_dir artifacts/runs/relation_prediction/wn18rr-rp \
  --train_paths artifacts/data/path_datasets/wn18rr/train/ \
  --valid_paths artifacts/data/path_datasets/wn18rr/val/ \
  --test_paths artifacts/data/path_datasets/wn18rr/test/ \
  --path_setup 20_20 --max_ppt 4 --batch_size 512 \
  --num_negatives 0 --val_num_negatives 0 \
  --nhead 2 --num_encoder_layers 2 \
  --ent_aggregation recurrent --num_agg_layers 1 \
  --num_agg_heads 1 --context_heads 0 \
  --dim_feedforward 128 --embedding_dim 64 \
  --lp_loss_fn ce --accumulate_gradient 8 \
  --monitor valid_loss --patience 10 --dropout 0.1 \
  --lrate 0.001 --weight_decay 0 --label_smoothing 0.1 \
  --device cuda --num_devices 1 --num_workers 0 --seed 46

python -m pathe.runner train pathe \
  --expname llf:ce_ea:avg_nal:1_nah:1_ch:0_bs:4096_mp:2_nn:0_nel:1_df:128_ed:32_ag:1_ls:0.01_cw_1 \
  --wandb_project wn18rr-rp \
  --log_dir artifacts/runs/relation_prediction/wn18rr-rp \
  --train_paths artifacts/data/path_datasets/wn18rr/train/ \
  --valid_paths artifacts/data/path_datasets/wn18rr/val/ \
  --test_paths artifacts/data/path_datasets/wn18rr/test/ \
  --path_setup 20_20 --max_ppt 2 --batch_size 1024 \
  --num_negatives 0 --val_num_negatives 0 \
  --nhead 2 --num_encoder_layers 1 \
  --ent_aggregation avg --num_agg_layers 1 \
  --num_agg_heads 1 --context_heads 0 \
  --dim_feedforward 128 --embedding_dim 32 \
  --lp_loss_fn ce --accumulate_gradient 8 \
  --monitor valid_loss --patience 10 --dropout 0.1 \
  --lrate 0.001 --weight_decay 0 --label_smoothing 0.01 \
  --device cuda --num_devices 1 --num_workers 0 --seed 46

python -m pathe.runner train pathe \
  --expname llf:ce_ea:recurrent_nal:2_nah:1_ch:1_bs:4096_d:0.2_mp:2_nn:0_n:4_nel:2_df:128_ed:32_ag:1_ls:0.01_cw_1837 \
  --wandb_project wn18rr-rp \
  --log_dir artifacts/runs/relation_prediction/wn18rr-rp \
  --train_paths artifacts/data/path_datasets/wn18rr/train/ \
  --valid_paths artifacts/data/path_datasets/wn18rr/val/ \
  --test_paths artifacts/data/path_datasets/wn18rr/test/ \
  --path_setup 20_20 --max_ppt 2 --batch_size 512 \
  --num_negatives 0 --val_num_negatives 0 \
  --nhead 4 --num_encoder_layers 2 \
  --ent_aggregation recurrent --num_agg_layers 2 \
  --num_agg_heads 1 --context_heads 1 \
  --dim_feedforward 128 --embedding_dim 32 \
  --lp_loss_fn ce --accumulate_gradient 8 \
  --monitor valid_loss --patience 10 --dropout 0.2 \
  --lrate 0.001 --weight_decay 0 --label_smoothing 0.01 \
  --device cuda --num_devices 1 --num_workers 0 --seed 46

python -m pathe.runner train pathe \
  --expname llf:ce_ea:transformer_nal:1_nah:1_ch:0_bs:4096_mp:4_nn:0_n:4_nel:2_df:128_ed:32_ag:1_ls:0.01_cw_221 \
  --wandb_project wn18rr-rp \
  --log_dir artifacts/runs/relation_prediction/wn18rr-rp \
  --train_paths artifacts/data/path_datasets/wn18rr/train/ \
  --valid_paths artifacts/data/path_datasets/wn18rr/val/ \
  --test_paths artifacts/data/path_datasets/wn18rr/test/ \
  --path_setup 20_20 --max_ppt 4 --batch_size 512 \
  --num_negatives 0 --val_num_negatives 0 \
  --nhead 4 --num_encoder_layers 2 \
  --ent_aggregation transformer --num_agg_layers 1 \
  --num_agg_heads 1 --context_heads 0 \
  --dim_feedforward 128 --embedding_dim 32 \
  --lp_loss_fn ce --accumulate_gradient 8 \
  --monitor valid_loss --patience 10 --dropout 0.1 \
  --lrate 0.001 --weight_decay 0 --label_smoothing 0.01 \
  --device cuda --num_devices 1 --num_workers 0 --seed 46

python -m pathe.runner train pathe \
  --expname llf:ce_ea:transformer_nal:1_nah:2_ch:2_bs:4096_d:0.2_mp:4_nn:0_nel:1_df:128_ed:64_ag:1_ls:0.1_546 \
  --wandb_project wn18rr-rp \
  --log_dir artifacts/runs/relation_prediction/wn18rr-rp \
  --train_paths artifacts/data/path_datasets/wn18rr/train/ \
  --valid_paths artifacts/data/path_datasets/wn18rr/val/ \
  --test_paths artifacts/data/path_datasets/wn18rr/test/ \
  --path_setup 20_20 --max_ppt 4 --batch_size 512 \
  --num_negatives 0 --val_num_negatives 0 \
  --nhead 2 --num_encoder_layers 1 \
  --ent_aggregation transformer --num_agg_layers 1 \
  --num_agg_heads 2 --context_heads 2 \
  --dim_feedforward 128 --embedding_dim 64 \
  --lp_loss_fn ce --accumulate_gradient 8 \
  --monitor valid_loss --patience 10 --dropout 0.2 \
  --lrate 0.001 --weight_decay 0 --label_smoothing 0.1 \
  --device cuda --num_devices 1 --num_workers 0 --seed 46
