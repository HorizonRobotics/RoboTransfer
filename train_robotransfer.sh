
export HF_DATASETS_CACHE=/horizon-bucket/robot_lab/users/jiagang.zhu/dataset/cache
export HF_ENDPOINT='https://hf-mirror.com'
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RDMAV_FORK_SAFE=1

pip install datasets
pip install wandb

accelerate launch  train_robotransfer.py \
  --pretrained_model_name_or_path="/horizon-bucket/robot_lab/users/jeff.wang/models/public/huggingface/models--stabilityai--stable-video-diffusion-img2vid-xt-1-1" \
  --train_batch_size=4 \
  --dataset_name "/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/agibot_prelabel_beta_0_2000_sorted/*.parquet"  \
  "/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/agibot_prelabel_beta_2000_4000_50/*.parquet" \
  "/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/agibot_prelabel_beta_4000_8000_100/*.parquet" \
  "/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/agibot_prelabel_beta_8000_12000_100/*.parquet" \
  "/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/agibot_prelabel_beta_12000_16000_100/*.parquet" \
  "/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/agibot_prelabel_beta_16000_20000_100/*.parquet" \
  "/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/agibot_prelabel_beta_20000_24000_100/*.parquet" \
  "/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/agibot_prelabel_beta_24000_28000_100/*.parquet" \
  "/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/agibot_prelabel_beta_28000_32000_100/*.parquet" \
  --bg_name '/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/fg_bg_total/fg_bg_0_2000' \
  '/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/fg_bg_total/fg_bg_2000_4000' \
  '/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/fg_bg_total/fg_bg_4000_8000' \
  '/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/fg_bg_total/fg_bg_8000_12000' \
  '/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/fg_bg_total/fg_bg_12000_16000' \
  '/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/fg_bg_total/fg_bg_16000_20000' \
  '/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/fg_bg_total/fg_bg_20000_24000' \
  '/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/fg_bg_total/fg_bg_24000_28000' \
  '/horizon-bucket/robot_lab/users/nemo.liu/datasets/AgiBotWorld/fg_bg_total/fg_bg_28000_32000' \
  --gradient_accumulation_steps=1 --gradient_checkpointing  \
  --num_train_epochs=30 \
  --learning_rate=3e-05 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --checkpointing_steps=1000 \
  --output_dir="/horizon-bucket/robot_lab/users/jiagang.zhu/checkpoint/nemo_sorted_3v_high" \
  --dataloader_num_workers 8 \
  --frame_num 8 \
  --width 640 \
  --height 384 \
  --multiview