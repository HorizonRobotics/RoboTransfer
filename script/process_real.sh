#!/bin/bash
# This script processes simulation data for mono normal and mono depth estimation.
# It takes a raw data directory and optional episode indices as arguments.
# Usage: ./process_sim.sh <raw_data_dir> [episode_idx1 episode_idx2 ...]
# Ensure the script is run with at least one argument

# Check if at least one argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <raw_data_dir> [episode_idx1 episode_idx2 ...]"
    exit 1
fi

# 1) Set the raw_data_dir and output_dir based on the provided arguments
raw_data_dir=$1  # the first argument is raw_data_dir
task_name=$(basename "$raw_data_dir")
output_dir="$PWD/data/real/${task_name}"

# remove the first argument, the remaining arguments are episode_idxs
shift
episode_idxs=("$@")  # 存储为数组
# 若未提供 episode_idxs，则默认为 (0)
if [ ${#episode_idxs[@]} -eq 0 ]; then
    episode_idxs=(0)
fi

echo episode_idxs: ${episode_idxs[@]}

# loop episode_idxs
for episode_idx in "${episode_idxs[@]}"; do
    echo "Dealing with episode_idx: $episode_idx"
    views=("left_camera" "middle_camera" "right_camera")
    episode_path="$raw_data_dir/$episode_idx"
    episode_output_dir="$output_dir/$episode_idx"

    # 1. load images
    echo "cpying episode directory from $episode_path to $episode_output_dir"
    cp -r "$episode_path" "$output_dir"

    # # 2.iterate over multiple views for mono normal
    # for view in "${views[@]}"; do
    #     view_dir="$episode_output_dir/$view/images"
    #     normal_output_dir="${episode_output_dir}/${view}/mono_normal"
    #     uv run script/run_mono_normal.py \
    #         --pretrained_model_name_or_path="jingheya/lotus-normal-g-v1-0" \
    #         --prediction_type="sample" \
    #         --task_name="normal" \
    #         --mode="generation" \
    #         --half_precision \
    #         --seed=42 \
    #         --input_dir="$view_dir" \
    #         --output_dir="$normal_output_dir"
    # done

    # 3.iterate over multiple views for mono depth
    for view in "${views[@]}"; do
        view_dir="$episode_output_dir/$view/images"
        depth_dir="$episode_output_dir/${view}/depth" #sensor_depth
        depth_output_dir="$episode_output_dir/${view}/mono_depth"
       
        uv run script/run_video_depth.py \
            --input_video $view_dir \
            --depth_dir $depth_dir \
            --output_dir  $depth_output_dir \
            --save_png \
            --save_vis \
            --target_fps 10 \
            # --no_depth_scaled #for data without sensor depth
   
    done
done