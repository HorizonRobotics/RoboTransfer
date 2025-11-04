#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import functools
import gc
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
import sys

import paths

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers.models import (
    UNetSpatioTemporalConditionModel,
    # AutoencoderKLTemporalDecoder
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


from transformers import CLIPVisionModelWithProjection
from einops import rearrange
import copy
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME
import safetensors

from robotransfer.loss.edm_loss import MVEDMLoss
from robotransfer.utils.image_loading import load_images_from_dataset
from robotransfer.utils.default_sampler import DefaultSampler
from robotransfer.utils.default_collator import DefaultCollator
from robotransfer.utils.lmdb_dataset import HuggingfaceNemoDataset
from robotransfer.models.guider import GuiderNet
from robotransfer.models.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
from torch.utils.data import BatchSampler

def load_state_dict(weight_path, weights_only=True):
    if os.path.isdir(weight_path):
        if os.path.exists(os.path.join(weight_path, WEIGHTS_NAME)):
            return torch.load(os.path.join(weight_path, WEIGHTS_NAME), map_location='cpu', weights_only=weights_only)
        elif os.path.exists(os.path.join(weight_path, SAFETENSORS_WEIGHTS_NAME)):
            return safetensors.torch.load_file(os.path.join(weight_path, SAFETENSORS_WEIGHTS_NAME), device='cpu')
        else:
            assert False
    elif os.path.isfile(weight_path):
        if weight_path.endswith('.safetensors'):
            return safetensors.torch.load_file(weight_path, device='cpu')
        else:
            return torch.load(weight_path, map_location='cpu', weights_only=weights_only)
    else:
        assert False

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.35.0.dev0")

logger = get_logger(__name__)
if is_torch_npu_available():
    import torch_npu

    torch.npu.config.allow_internal_format = False

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        nargs='*',
        default=[],
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--bg_name",
        nargs='*',
        default=[],
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=192,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sdxl-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--frame_num", type=int, default=30, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--multiview",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--timestep_bias_strategy",
        type=str,
        default="none",
        choices=["earlier", "later", "range", "none"],
        help=(
            "The timestep bias strategy, which may help direct the model toward learning low or high frequency details."
            " Choices: ['earlier', 'later', 'range', 'none']."
            " The default is 'none', which means no bias is applied, and training proceeds normally."
            " The value of 'later' will increase the frequency of the model's final training timesteps."
        ),
    )
    parser.add_argument(
        "--timestep_bias_multiplier",
        type=float,
        default=1.0,
        help=(
            "The multiplier for the bias. Defaults to 1.0, which means no bias is applied."
            " A value of 2.0 will double the weight of the bias, and a value of 0.5 will halve it."
        ),
    )
    parser.add_argument(
        "--timestep_bias_begin",
        type=int,
        default=0,
        help=(
            "When using `--timestep_bias_strategy=range`, the beginning (inclusive) timestep to bias."
            " Defaults to zero, which equates to having no specific bias."
        ),
    )
    parser.add_argument(
        "--timestep_bias_end",
        type=int,
        default=1000,
        help=(
            "When using `--timestep_bias_strategy=range`, the final timestep (inclusive) to bias."
            " Defaults to 1000, which is the number of timesteps that Stable Diffusion is trained on."
        ),
    )
    parser.add_argument(
        "--timestep_bias_portion",
        type=float,
        default=0.25,
        help=(
            "The portion of timesteps to bias. Defaults to 0.25, which 25% of timesteps will be biased."
            " A value of 0.5 will bias one half of the timesteps. The value provided for `--timestep_bias_strategy` determines"
            " whether the biased portions are in the earlier or later timesteps."
        ),
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://huggingface.co/papers/2303.09556.",
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_npu_flash_attention", action="store_true", help="Whether or not to use npu flash attention."
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--image_interpolation_mode",
        type=str,
        default="lanczos",
        choices=[
            f.lower() for f in dir(transforms.InterpolationMode) if not f.startswith("__") and not f.endswith("__")
        ],
        help="The image interpolation method to use for resizing images.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    return args


def process_unet(unet, latent_channels, unet_cfg):
    in_channels = unet_cfg.get('in_channels', latent_channels * 2)
    out_channels = unet_cfg.get('out_channels', latent_channels)
    if unet.config.in_channels != in_channels or unet.config.out_channels != out_channels:
        unet_config = copy.deepcopy(unet.config)
        unet_config['in_channels'] = in_channels
        unet_config['out_channels'] = out_channels
        new_unet = UNetSpatioTemporalConditionModel(**unet_config)
        state_dict = unet.state_dict()
        if unet.config.in_channels != in_channels:
            weight = state_dict['conv_in.weight']
            if unet.config.in_channels > in_channels:
                new_weight = weight[:, :in_channels]
            else:
                weight_shape = list(weight.shape)
                weight_shape[1] = in_channels
                new_weight = weight.new_zeros(weight_shape)
                new_weight[:, : weight.shape[1]] = weight
                # state_dict.pop('conv_in.weight')
                # state_dict.pop('conv_in.bias')
            state_dict['conv_in.weight'] = new_weight
        if unet.config.out_channels != out_channels:
            state_dict.pop('conv_out.weight')
            state_dict.pop('conv_out.bias')
        new_unet.load_state_dict(state_dict, strict=False)
        del unet
        unet = new_unet
    return unet

def get_dataloaders(args):
    
    batch_size_per_gpu = args.train_batch_size
    multiview = args.multiview
    frame_num = args.frame_num

    dataset_name = args.dataset_name
    bg_name =  args.bg_name
    width = args.width
    height = args.height
    
    dataset = HuggingfaceNemoDataset(dataset_name, bg_name=bg_name, multiview=multiview, frame_num=frame_num, target_size=(width, height))
    sampler = DefaultSampler(dataset=dataset, batch_size=batch_size_per_gpu)
    batch_sampler = BatchSampler(sampler, batch_size=batch_size_per_gpu, drop_last=False)
    
    print("dataloader_num_workers: ", args.dataloader_num_workers)

    collator = DefaultCollator()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        num_workers=args.dataloader_num_workers
    )
    print("len(dataset): ", len(dataset))
    return dataloader, len(dataset)

def main(args):

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    edm=dict(
            sigma_method=2,
            p_mean=1.0,
            p_std=1.6,
            sigma_data=-1,
            cam_keys=['concat'],
        )
    edm_loss = MVEDMLoss(**edm)
    # Check for terminal SNR in combination with SNR Gamma

    vae_path = (
        args.pretrained_model_name_or_path
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        vae_path,
        subfolder="vae",
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    latent_channels = vae.config.latent_channels
    unet_cfg = dict(
            in_channels=8 + 4 + 4,
            )
    unet = process_unet(unet, latent_channels, unet_cfg)

    # Freeze vae and text encoders.
    vae.enable_slicing()  ### memory reduction
    vae.requires_grad_(False)

    # Set unet as trainable.
    unet.train()

    depth_guider_cfg=dict(
        in_channels=1,
        out_channels=4,
    )
    normal_guider_cfg=dict(
        in_channels=3,
        out_channels=4,
    )

    depth_guider_net = GuiderNet(**depth_guider_cfg)
    depth_guider_net.train()

    normal_guider_net = GuiderNet(**normal_guider_cfg)
    normal_guider_net.train()

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder='image_encoder')
    image_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=torch.float32)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNetSpatioTemporalConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNetSpatioTemporalConditionModel, model_config=ema_unet.config)
    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            unet.enable_npu_flash_attention()
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu devices.")
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    
    save_models_names = ['unet', 'depth_guider_net', 'normal_guider_net']
    
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    
                    print(f"saving model {i}", os.path.join(output_dir, save_models_names[i]))
                    
                    model.save_pretrained(os.path.join(output_dir, save_models_names[i]))

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNetSpatioTemporalConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model
                
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                save_models_name_tmp = save_models_names.pop()

                # load diffusers style into model
                if 'unet' in save_models_name_tmp:
                    load_model = UNetSpatioTemporalConditionModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model
                elif 'depth' in  save_models_name_tmp:
                    state_dict = load_state_dict(os.path.join(input_dir, save_models_name_tmp))
                    model.load_state_dict(state_dict)
                else:
                    state_dict = load_state_dict(os.path.join(input_dir, save_models_name_tmp))
                    model.load_state_dict(state_dict)                   
                    
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    def _apply_activation_checkpointing(models=None):
        activation_class_names=['BasicTransformerBlock', 'TemporalBasicTransformerBlock', 'SpatioTemporalResBlock']
        if activation_class_names is not None:
            for model in models:
                cls_to_wrap = set()
                for class_name in activation_class_names:
                    for module in model.modules():
                        if module.__class__.__name__ == class_name:
                            cls_to_wrap.add(module.__class__)
                            break
                auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=cls_to_wrap)
                apply_activation_checkpointing(model, auto_wrap_policy=auto_wrap_policy)


    if args.gradient_checkpointing:
        # unet.enable_gradient_checkpointing()   ### less memory reduction
        _apply_activation_checkpointing([unet])  ### more memory reduction
        unet = unet
        
        

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    # params_to_optimize = unet.parameters()
    params_to_optimize = (
        list(unet.parameters()) 
        + list(depth_guider_net.parameters()) 
        + list(normal_guider_net.parameters())
    )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
    train_dataloader, num_samples = get_dataloaders(args)
    
    gc.collect()
    if is_torch_npu_available():
        torch_npu.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    if args.max_train_steps is None:
        overrode_max_train_steps = True

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) // accelerator.num_processes
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, depth_guider_net, normal_guider_net, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, depth_guider_net, normal_guider_net, optimizer, train_dataloader, lr_scheduler
    )
    
    models = [unet, depth_guider_net, normal_guider_net]

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune-sdxl", config=vars(args))

    # Function for unwrapping if torch.compile() was used in accelerate.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if torch.backends.mps.is_available() or "playground" in args.pretrained_model_name_or_path:
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_samples}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = args.resume_from_checkpoint
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                args.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(args.output_dir, path))
                global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    fps = 6
    motion_bucket_id = 127
    noise_aug_strength = 0.0
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        dataloader_iter = iter(train_dataloader)
        for i in range(num_update_steps_per_epoch):
            batch_dict = next(dataloader_iter)
            with accelerator.accumulate(*models):
                # Sample noise that we'll add to the latents
                image = batch_dict['concat'].to(vae.dtype)
                
                image = image.to(vae.device)                
                batch_size, num_frames = image.shape[:2]
                lossw = []
                for bz in range(batch_size):
                    if torch.max(image[bz]) <=-1.0:
                        # print("image: ", image.shape)
                        print("lossw: 0", 0)
                        lossw.append(0.0)
                        
                    else:
                        lossw.append(1.0)
                        
                with torch.no_grad():
                    image = image.flatten(0, 1)
                    latents = vae.encode(image).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                cam_latents = {}
                cam_latents['concat'] = latents
                # add noise
                inp_noisy_latents, timesteps = edm_loss.add_noise(cam_latents, batch_size)
                timesteps = timesteps.to(dtype=weight_dtype)

                # added_time_ids
                added_time_ids = [fps, motion_bucket_id, noise_aug_strength]
                added_time_ids = torch.tensor([added_time_ids], device=accelerator.device)
                added_time_ids = torch.cat([added_time_ids] * batch_size)

                # conditional_latents
                ref_image =  batch_dict['concat_ref_image'].to(vae.dtype)

                with torch.no_grad():
                    ref_image = ref_image.flatten(0, 1)
                    conditional_latents = vae.encode(ref_image).latent_dist.sample()
                
                ref_drop_prob = 0.1
                if ref_drop_prob > 0:
                    random_p = torch.rand(batch_size, device=accelerator.device)
                    image_mask = random_p <= ref_drop_prob
                    image_mask = 1 - image_mask.float()
                    image_mask = image_mask.to(conditional_latents.dtype)
                    conditional_latents = conditional_latents * image_mask[:, None, None, None]
                conditional_latents = conditional_latents.repeat_interleave(num_frames, dim=0)
                
                with accelerator.autocast():
                    depth_guider_images = batch_dict['concat_depth'].to(weight_dtype)
                    depth_guider_images = depth_guider_images.flatten(0, 1)
                    depth_guider_images_latents = depth_guider_net(depth_guider_images)
                    
                    normal_guider_images = batch_dict['concat_normal'].to(weight_dtype)
                    normal_guider_images = normal_guider_images.flatten(0, 1)
                    normal_guider_images_latents = normal_guider_net(normal_guider_images)
                    
                    inp_noisy_latents['concat'] = torch.cat([inp_noisy_latents['concat'], conditional_latents], dim=1)                
                    inp_noisy_latents['concat'] = torch.cat([inp_noisy_latents['concat'], depth_guider_images_latents], dim=1)
                    inp_noisy_latents['concat'] = torch.cat([inp_noisy_latents['concat'], normal_guider_images_latents], dim=1)

                    embeddings = {}
                    clip_image = batch_dict['concat_clip_image'].flatten(0, 1).to(weight_dtype)
                    obj_clip_image = batch_dict['concat_obj_clip_image'].flatten(0, 1).to(weight_dtype)
                    
                    with torch.no_grad():
                        clip_embedding = image_encoder(clip_image).image_embeds.unsqueeze(1)
                        obj_clip_embedding = image_encoder(obj_clip_image).image_embeds.unsqueeze(1)
                        feat_dim = obj_clip_embedding.shape[-1]
                        obj_clip_embedding = obj_clip_embedding.view(batch_size, -1, feat_dim)                        
                        embeddings['concat'] = torch.cat([clip_embedding, obj_clip_embedding], dim=1).to(dtype=weight_dtype)
                    
                    if ref_drop_prob > 0:
                        image_mask = random_p <= ref_drop_prob
                        image_mask = 1 - image_mask.float()
                        image_mask = image_mask.to(embeddings['concat'].dtype)
                        embeddings['concat'] = embeddings['concat'] * image_mask[:, None, None]

                    mv_inp_noisy_latents = inp_noisy_latents['concat'].to(dtype=weight_dtype)
                    mv_embeddings = embeddings['concat'].to(dtype=weight_dtype)
 
                    mv_inp_noisy_latents = rearrange(mv_inp_noisy_latents, "(b f) c h w -> b f c h w", b=batch_size)
                    added_time_ids = added_time_ids.to(dtype=weight_dtype)
                    encoder_attention_mask = batch_dict['encoder_attention_mask'].to(dtype=weight_dtype)
                    
                    model_pred = unet(
                        mv_inp_noisy_latents,
                        timesteps,
                        encoder_hidden_states=mv_embeddings,
                        encoder_attention_mask=encoder_attention_mask,
                        added_time_ids=added_time_ids,
                    )[0]
                    
                    model_pred={
                        'concat': model_pred
                    }
            
                # compute loss
                denoised_latents = edm_loss.denoise(model_pred)
                loss = edm_loss.compute_loss(denoised_latents)
                
                # print("loss: ", loss.shape)
                for bz in range(batch_size):
                    loss[bz] = loss[bz] * lossw[bz]
                    

                loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = unet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if args.use_ema:
                        ema_unet.step(unet.parameters())
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                    # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                    if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                        if global_step % args.checkpointing_steps == 0:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(args.output_dir, f"checkpoint-epoch{epoch}-step{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)