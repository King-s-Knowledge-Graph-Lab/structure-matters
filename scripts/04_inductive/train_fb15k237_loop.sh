#!/bin/bash

# Define the range of the counter
# start=1
# end=4

# run from the repository root with the PathE environment activated and pathe installed (pip install -e .)

start=1
end=4

# Iterate over the counter
for ((counter=start; counter<=end; counter++))
do
    # Define variables to pass to the Python script
    var1="v$counter"


    
    # Run the Python script with the variables
    python -m pathe.runner train pathe --log_dir artifacts/runs/inductive/fb15k237/"$var1" \
    --expname \
    l2-h2-e64-ls0001-d1-ag8-n299-dummy-avg-1l-lr1e3-"$var1" \
    --wandb_project \
    fb15k237-ind \
    --train_paths \
    artifacts/data/path_datasets/fb15k237/"$var1"/train/ \
    --valid_paths \
    artifacts/data/path_datasets/fb15k237/"$var1"/val/ \
    --test_paths \
    artifacts/data/path_datasets/fb15k237/"$var1"/test/ \
    --inference_paths \
    artifacts/data/path_datasets/fb15k237/"$var1"/inference/ \
    --max_ppt 2 \
    --batch_size 1024\
    --nhead 2 \
    --num_encoder_layers 1 \
    --dim_feedforward 256 \
    --embedding_dim 64 \
    --num_negatives 299 \
    --monitor valid_link_hits10 \
    --patience 5 \
    --dropout 0.1 \
    --lrate 1e-3 \
    --device cuda \
    --num_devices 1 \
    --val_num_negatives 50\
    --num_workers 10 \
    --ent_aggregation avg \
    --lp_loss_fn ce \
    --accumulate_gradient 8 \
    --label_smoothing 0.01 \
    --node_projector dummy \
    --path_setup 20_10 \
    --num_agg_layers 1 \
    --seed 7
    
    # Optional: Add a delay between iterations
    # sleep 1
done
