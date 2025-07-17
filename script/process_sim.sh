#!/bin/bash
# This script processes simulation data for mono normal estimation.
# It takes a raw data directory and optional episode indices as arguments.
# Usage: ./process_sim.sh <raw_data_dir> [episode_idx1 episode_idx2 ...]

# Ensure the script is run with at least one argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 <raw_data_dir> [episode_idx1 episode_idx2 ...]"
    exit 1
fi

# Set the raw_data_dir and output_dir based on the provided arguments
raw_data_dir=$1  # the first argument is raw_data_dir
echo "raw_data_dir: $raw_data_dir"

task_name=$(basename "$raw_data_dir")
output_dir="$PWD/data/sim/${task_name}"
echo "output_dir: $output_dir"

# Remove the first argument, the remaining arguments are episode_idxs
shift
episode_idxs=("$@")  #array
# if no episode_idxs are provided, default to (0)
if [ ${#episode_idxs[@]} -eq 0 ]; then
    episode_idxs=(0)
fi

echo episode_idxs: ${episode_idxs[@]}

# loop episode_idxs
for episode_idx in "${episode_idxs[@]}"; do
    views=("left_camera" "middle_camera" "right_camera")

    # Set the episode path based on the file or directory structure
    if [ -f "$raw_data_dir/episode${episode_idx}.hdf5" ]; then
        # If the file exists, use it as the episode path
        episode_path="$raw_data_dir/episode${episode_idx}.hdf5"
        echo "Found HDF5 file: $episode_path"
    # Then check for a directory with pickle files
    elif [ -d "$raw_data_dir/episode${episode_idx}" ]; then
        # Assuming the directory contains pickle files
        episode_path="$raw_data_dir/episode${episode_idx}"
        echo "Found pickle directory: $episode_path"
    else
        echo "Error: Episode $episode_idx not found as a .hdf5 file or a pickle directory in $raw_data_dir"
        exit 1 # Exit with an error code
    fi


    # Now you can use $episode_path in your script
    echo "Processing episode path: $episode_path"
    episode_output_dir="$output_dir/episode$episode_idx"
    echo "Output directory for episode: $episode_output_dir"

    # load images from pickle
    uv run script/sim2images.py \
        --episode_path="$episode_path" \
        --output_dir="$output_dir"
    
    # iterate over multiple views for mono normal
    for view in "${views[@]}"; do
        view_dir="$episode_output_dir/$view/rgb"
        normal_output_dir="${episode_output_dir}/${view}/mono_normal"

        uv run script/run_mono_normal.py \
            --pretrained_model_name_or_path="jingheya/lotus-normal-g-v1-0" \
            --prediction_type="sample" \
            --task_name="normal" \
            --mode="generation" \
            --half_precision \
            --seed=42 \
            --input_dir="$view_dir" \
            --output_dir="$normal_output_dir"
    done
    echo "Finished processing mono normal for episode: $episode_idx"
done
