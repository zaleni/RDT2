#!/usr/bin/env python
# coding=utf-8
import logging
import math
import os
import socket
from pathlib import Path
from functools import partial
from itertools import chain  # <-- add

import diffusers
import torch
import torch.nn as nn
import transformers
import yaml
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, DataLoaderConfiguration, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# from models.multimodal_encoder.dinosiglip_vit import DinoSigLIPViTBackbone
from models.normalizer import LinearNormalizer
from models.rdt_runner import RDTRunner
from rdt.dataset import get_instructions_and_blended_train_dataset, collate_fn
# from rdt.models.rdt_distill_runner import RDTDistillRunner
from rdt.sample import log_sample_res


if is_wandb_available():
    import wandb

class _RDTWithVLM(nn.Module):
    """DeepSpeed 只允许一个 model：把 RDT + VLM 封装成单一 nn.Module。"""
    def __init__(self, rdt: nn.Module, vlm: nn.Module, selected_layers):
        super().__init__()
        self.rdt = rdt
        self.vlm = vlm
        self.selected_layers = selected_layers

    def forward(
        self,
        *,
        vlm_inputs: dict,
        lang_attn_mask: torch.Tensor,
        image_embeds: torch.Tensor | None,
        states: torch.Tensor,
        nsamples: torch.Tensor,
    ):
        # 如果 VLM 冻结，用 no_grad 省显存
        vlm_trainable = any(p.requires_grad for p in self.vlm.parameters())
        if (not self.vlm.training) or (not vlm_trainable):
            with torch.no_grad():
                outputs = self.vlm(**vlm_inputs, use_cache=True)
        else:
            outputs = self.vlm(**vlm_inputs, use_cache=True)

        pkv = outputs.past_key_values
        if isinstance(self.selected_layers, list):
            vlang_kv_cache = [pkv[i] for i in self.selected_layers]
        else:
            vlang_kv_cache = [pkv[self.selected_layers]]

        loss = self.rdt(
            lang_kv_cache=vlang_kv_cache,
            lang_attn_mask=lang_attn_mask,
            img_tokens=image_embeds,
            state_tokens=states,
            action_gt=nsamples,
        )
        return loss

def save_model_card(repo_id: str, base_model=str, repo_folder=None):
    yaml = f"""
---
license: mit
base_model: {base_model}
language:
- en
pipeline_tag: robotics
library_name: transformers
tags:
- robotics
- pytorch
- multimodal
- pretraining
- vla
- diffusion
- rdt
---
    """
    model_card = f"""
# RDT - {repo_id}

This is a RDT model derived from {base_model}. The weights were trained using [RDT](https://rdt-robotics.github.io/rdt-robotics/).
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def train(args, logger):
    # Read the config
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(
            dispatch_batches=(
                False if args.hdf5_dir is None else None
            ),
        ),
        deepspeed_plugin=DeepSpeedPlugin(
            hf_ds_config=args.deepspeed
        ) if args.deepspeed is not None else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
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

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    processor = AutoProcessor.from_pretrained(
        args.pretrained_vision_language_model_name_or_path,
        padding_side="left",
        use_fast=True,
    )
    vision_language_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.pretrained_vision_language_model_name_or_path,
        torch_dtype=weight_dtype,
        attn_implementation="flash_attention_2",
        device_map=None,
    )

    if args.train_vlm:
        vision_language_model.train()
        vision_language_model.requires_grad_(True)
    else:
        vision_language_model.eval()
        vision_language_model.requires_grad_(False)

    # tokenizer = processor.tokenizer

    # if (
    #     config["model"].get("img_adaptor", None) is not None 
    #     and args.pretrained_vision_encoder_name_or_path is not None
    # ):
    #     vision_encoder = DinoSigLIPViTBackbone(
    #         vision_backbone_id=args.pretrained_vision_encoder_name_or_path,
    #         image_resize_strategy="letterbox"
    #             if config["dataset"]["image_aspect_ratio"] == "pad"
    #             else "resize-naive",
    #         default_image_size=384
    #     )
    #     image_transform = vision_encoder.get_image_transform()
    # else:
    vision_encoder = image_transform = None

    if isinstance(config["model"]["selected_layers"], list):
        assert len(config["model"]["selected_layers"]) == config["model"]["rdt"]["depth"], (
            f"selected_layers ({config['model']['selected_layers']}) must be a list of integers with length "
            f"{config['model']['rdt']['depth']}"
        )
    elif not isinstance(config["model"]["selected_layers"], int):
        raise ValueError(
            f"selected_layers ({config['model']['selected_layers']}) must be a list of integers or an integer"
        )

    # Create RDT config
    rdt_config = {
        "state_dim": config["common"]["state_dim"],
        "action_dim": config["common"]["action_dim"],
        "pred_horizon": config["common"]["action_chunk_size"],
        "config": config["model"],
        "act_pos_emb_config": [
            ('action', config["common"]["action_chunk_size"]),
            # Learnable register tokens
            ('register', config["model"]["rdt"]["num_register_tokens"]),
        ],
        "dtype": weight_dtype,
    }
    
    if config["model"].get("lang_adaptor", None) is not None:
        rdt_config.update({
            "lang_token_dim": config["model"]["lang_token_dim"],
        })
    
    if config["model"].get("img_adaptor", None) is not None:
        assert vision_encoder is not None, \
            "Vision encoder must be provided when img_adaptor is used."
        rdt_config.update({
            "img_token_dim": config["model"]["img_token_dim"],
            "img_pos_emb_config": [
            # No initial pos embed in the last grid size
            # since we've already done in ViT
            ("image", (config["common"]["img_history_size"],
                       config["common"]["num_cameras"],
                       -vision_encoder.num_patches)),
            ],
            "max_img_len": (
                config["common"]["img_history_size"]
                * config["common"]["num_cameras"]
                * vision_encoder.num_patches
            ),
        
        })
    
    # Load from a pretrained checkpoint
    if (
        args.pretrained_model_name_or_path is not None
        and not os.path.isfile(args.pretrained_model_name_or_path)
    ):
        logger.info("Loading model from a pretrained checkpoint directory.")
        rdt = RDTRunner.from_pretrained(args.pretrained_model_name_or_path)
    else:
        rdt = RDTRunner(**rdt_config)

        if (
            args.resume_from_checkpoint is None
            and args.pretrained_model_name_or_path is not None
            and os.path.isfile(args.pretrained_model_name_or_path)
        ):
            # Since EMA is deprecated, we do not load EMA from the pretrained checkpoint
            logger.info("Loading model from a pretrained checkpoint file.")
            checkpoint = torch.load(args.pretrained_model_name_or_path)
            rdt.load_state_dict(checkpoint["module"])
        else:
            logger.info("Constructing model from scratch.")

    # Initialize the distillation model
    if args.enable_distill:
        if args.pretrained_model_name_or_path is None:
            raise ValueError("When distillation is enabled, `pretrained_model_name_or_path` must be provided.")
        logger.info("Constructing distillation model from pretrained checkpoint.")
        rdt_ditill = RDTDistillRunner(
            rdt_config=rdt_config,
            dtype=weight_dtype,
        )
        rdt_ditill.initialize_from_pretrained_denoiser(rdt)
        rdt = rdt_ditill    # in-place replace the original RDT with the distillation model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    # which ensure saving model in huggingface format (config.json + pytorch_model.bin)
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return

        for model in models:
            model_to_save = model.module if hasattr(model, "module") else model  # type: ignore

            # 兼容：单独 rdt 或 wrapper
            if isinstance(model_to_save, _RDTWithVLM):
                # Save RDT
                model_to_save.rdt.save_pretrained(output_dir)
                # Save VLM (only if we finetune it)
                if args.train_vlm:
                    vlm_dir = os.path.join(output_dir, "vlm")
                    os.makedirs(vlm_dir, exist_ok=True)
                    model_to_save.vlm.save_pretrained(vlm_dir)
                    processor.save_pretrained(vlm_dir)

            else:
                # Save RDT
                if isinstance(model_to_save, type(accelerator.unwrap_model(rdt))):
                    model_to_save.save_pretrained(output_dir)

                # Save VLM (only if we finetune it)
                if args.train_vlm and isinstance(model_to_save, Qwen2_5_VLForConditionalGeneration):
                    vlm_dir = os.path.join(output_dir, "vlm")
                    os.makedirs(vlm_dir, exist_ok=True)
                    model_to_save.save_pretrained(vlm_dir)
                    processor.save_pretrained(vlm_dir)

    accelerator.register_save_state_pre_hook(save_model_hook)

    if args.gradient_checkpointing:
        # TODO:
        raise NotImplementedError("Gradient checkpointing is not yet implemented.")

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
    if args.train_vlm:
        params_to_optimize = [
            {"params": rdt.parameters(), "lr": args.learning_rate, "weight_decay": args.adam_weight_decay},
            {"params": vision_language_model.parameters(), "lr": args.vlm_learning_rate, "weight_decay": args.vlm_weight_decay},
        ]
    else:
        params_to_optimize = rdt.parameters()

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    assert args.hdf5_dir is not None or args.webdataset_config is not None, \
        "Either hdf5_dir or webdataset_config must be provided."
    assert not (args.hdf5_dir is not None and args.webdataset_config is not None), \
        "Currently only one of hdf5_dir or webdataset_config can be provided."

    # if args.hdf5_dir is not None:
    #     # Dataset and DataLoaders creation:
    #     train_dataset = H5VLADataset(
    #         config=config,
    #         hdf5_dir=args.hdf5_dir,
    #         tokenizer=processor,
    #         image_transform=image_transform,
    #         num_cameras=config["common"]["num_cameras"],
    #         img_history_size=config["common"]["img_history_size"],
    #         auto_adjust_image_brightness=args.auto_adjust_image_brightness,
    #         image_aug=args.image_aug,
    #         cond_mask_prob=args.cond_mask_prob,
    #         cam_ext_mask_prob=args.cam_ext_mask_prob,
    #         state_noise_snr=args.state_noise_snr,
    #         use_precomp_lang_embed=args.precomp_lang_embed
    #     )
    #     sample_dataset = H5VLADataset(
    #         config=config,
    #         hdf5_dir=args.hdf5_dir,
    #         tokenizer=processor,
    #         image_transform=image_transform,
    #         num_cameras=config["common"]["num_cameras"],
    #         img_history_size=config["common"]["img_history_size"],
    #         auto_adjust_image_brightness=args.auto_adjust_image_brightness,
    #         image_aug=False,
    #         cond_mask_prob=0,
    #         cam_ext_mask_prob=-1,
    #         state_noise_snr=None,
    #         use_precomp_lang_embed=args.precomp_lang_embed
    #     )

    #     train_dataloader = torch.utils.data.DataLoader(
    #         train_dataset,
    #         batch_size=args.train_batch_size,
    #         shuffle=True,
    #         collate_fn=train_dataset._collate_fn,
    #         num_workers=args.dataloader_num_workers,
    #         pin_memory=True,
    #         persistent_workers=True
    #     )
    #     sample_dataloader = torch.utils.data.DataLoader(
    #         sample_dataset,
    #         batch_size=args.sample_batch_size,
    #         shuffle=True,
    #         collate_fn=sample_dataset._collate_fn,
    #         num_workers=args.dataloader_num_workers,
    #         pin_memory=True,
    #         persistent_workers=True
    #     )
    # else:
    with open(args.webdataset_config, "r") as f:
        hostname = socket.gethostname()
        dataset_config_str = f.read().format(hostname=hostname)
        wds_config = yaml.safe_load(dataset_config_str)

    instructions, train_dataset = get_instructions_and_blended_train_dataset(wds_config)
        
    train_collate_fn = partial(
        collate_fn, 
        processor=processor,
        instructions=instructions, 
        image_corruption=args.image_aug,
        state_dim=config["common"]["state_dim"]
    )
    
    # TODO(lingxuan): use unique dataset for validation
    sample_dataloader = train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        collate_fn=train_collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    normalizer = LinearNormalizer.load(wds_config["kwargs"]["normalizer_path"])

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    if args.max_train_steps is None:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    # DeepSpeed 下 train_vlm=True 时只能 prepare 一个 model：用 wrapper
    using_ds = args.deepspeed is not None

    if args.train_vlm and using_ds:
        train_model = _RDTWithVLM(rdt=rdt, vlm=vision_language_model, selected_layers=config["model"]["selected_layers"])
        train_model, optimizer, train_dataloader, sample_dataloader, lr_scheduler = accelerator.prepare(
            train_model, optimizer, train_dataloader, sample_dataloader, lr_scheduler
        )
        # 方便后面继续使用原变量名（注意：不要单独对 vlm 做 forward 训练；训练走 train_model）
        inner = train_model.module if hasattr(train_model, "module") else train_model
        rdt = inner.rdt
        vision_language_model = inner.vlm
    else:
        if args.train_vlm:
            rdt, vision_language_model, optimizer, train_dataloader, sample_dataloader, lr_scheduler = accelerator.prepare(
                rdt, vision_language_model, optimizer, train_dataloader, sample_dataloader, lr_scheduler
            )
        else:
            rdt, optimizer, train_dataloader, sample_dataloader, lr_scheduler = accelerator.prepare(
                rdt, optimizer, train_dataloader, sample_dataloader, lr_scheduler
            )
            vision_language_model.to(accelerator.device, dtype=weight_dtype)
        train_model = None  # <-- add

    if vision_encoder is not None:
        vision_encoder.to(accelerator.device, dtype=weight_dtype)


    # 不要在 train_vlm=True 且 prepare 之后再强行 .to()，避免对 deepspeed/包裹对象做多余操作
    
    # if vision_language_model is not None:
    #     vision_language_model.to(accelerator.device, dtype=weight_dtype)

    if overrode_max_train_steps:
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        if hasattr(train_dataset, "__len__") and len(train_dataset) > 0:
            num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        else:
            num_update_steps_per_epoch = args.max_train_steps
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        
    # Afterwards we recalculate our number of training epochs
    # args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # accelerator.init_trackers("rdt-2", config=vars(args))
        accelerator.init_trackers(os.getenv("WANDB_PROJECT", "rdt-2"), config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    # logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            try:
                accelerator.load_state(os.path.join(args.output_dir, path))
            except:
                logger.info("Resuming training state failed. Attempting to only load from model checkpoint.")
                checkpoint = torch.load(os.path.join(args.output_dir, path, "pytorch_model", "mp_rank_00_model_states.pt"))

                # 兼容 wrapper / 非 wrapper
                if args.train_vlm and using_ds and train_model is not None:
                    target = train_model.module if hasattr(train_model, "module") else train_model
                else:
                    target = rdt.module if hasattr(rdt, "module") else rdt
                target.load_state_dict(checkpoint.get("module", checkpoint), strict=False)

            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        rdt.train()
        if args.train_vlm:
            vision_language_model.train()
        else:
            vision_language_model.eval()

        # Set the progress_bar to correct position
        if args.resume_from_checkpoint and epoch == first_epoch:
            progress_bar.update(resume_step // args.gradient_accumulation_steps)

        # Forward and backward...
        for batch in train_dataloader:
            # DeepSpeed+train_vlm：accumulate 只传单个 wrapper model
            if args.train_vlm and using_ds and train_model is not None:
                accumulate_ctx = accelerator.accumulate(train_model)
            else:
                accumulate_ctx = accelerator.accumulate(rdt)

            with accumulate_ctx:
                actions = batch["actions"]
                nsamples = normalizer["action"].normalize(actions).to(
                    dtype=weight_dtype, device=accelerator.device
                )

                states = batch["states"].to(dtype=weight_dtype, device=accelerator.device)

                with torch.no_grad():
                    image_embeds = None
                    if vision_encoder is not None:
                        images = {k: v.to(device=accelerator.device, dtype=weight_dtype) for k, v in batch["images"].items()}
                        k = next(iter(images))
                        batch_size, _, C, H, W = images[k].shape
                        for k in images:
                            images[k] = images[k].reshape(-1, C, H, W)
                        image_embeds = vision_encoder(images).detach()
                        image_embeds = image_embeds.reshape((batch_size, -1, vision_encoder.embed_dim))

                vlm_inputs = batch["vision_language_model_inputs"]
                vlm_inputs = {
                    k: (
                        v.to(device=accelerator.device, dtype=weight_dtype)
                        if torch.is_tensor(v) and v.is_floating_point()
                        else (v.to(device=accelerator.device) if torch.is_tensor(v) else v)
                    )
                    for k, v in vlm_inputs.items()
                }
                lang_attn_mask = vlm_inputs["attention_mask"].to(dtype=torch.bool)

                # 训练 loss：DeepSpeed+train_vlm 走 wrapper forward（单模型）
                if args.train_vlm and using_ds and train_model is not None:
                    loss = train_model(
                        vlm_inputs=vlm_inputs,
                        lang_attn_mask=lang_attn_mask,
                        image_embeds=image_embeds,
                        states=states,
                        nsamples=nsamples,
                    )
                else:
                    # ---- 原逻辑保持：VLM 前向只跑一次 ----
                    if args.train_vlm:
                        outputs = vision_language_model(**vlm_inputs, use_cache=True)
                    else:
                        with torch.no_grad():
                            outputs = vision_language_model(**vlm_inputs, use_cache=True)

                    selected_layers = config["model"]["selected_layers"]
                    if isinstance(selected_layers, list):
                        vlang_kv_cache = [outputs.past_key_values[i] for i in selected_layers]
                    else:
                        vlang_kv_cache = [outputs.past_key_values[selected_layers]]

                    loss = rdt(
                        lang_kv_cache=vlang_kv_cache,
                        lang_attn_mask=lang_attn_mask,
                        img_tokens=image_embeds,
                        state_tokens=states,
                        action_gt=nsamples,
                    )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if args.train_vlm:
                        if args.train_vlm and using_ds and train_model is not None:
                            params_to_clip = train_model.parameters()
                        else:
                            params_to_clip = chain(rdt.parameters(), vision_language_model.parameters())
                    else:
                        params_to_clip = rdt.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_period == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                if args.sample_period > 0 and global_step % args.sample_period == 0:
                    sample_loss_for_log = log_sample_res(
                        vision_language_model,
                        config["model"]["selected_layers"],
                        vision_encoder,
                        rdt,    # We do not use EMA currently
                        normalizer,
                        args,
                        accelerator,
                        weight_dtype,
                        sample_dataloader,
                        logger,
                    )
                    logger.info(sample_loss_for_log)
                    accelerator.log(sample_loss_for_log, step=global_step)

            logs = {"loss": loss.detach().item()}
            
            # 自动遍历优化器里的所有参数组
            for i, pg in enumerate(optimizer.param_groups):
                # 如果是第0组通常是RDT，第1组是VLM (根据你创建optimizer时的顺序)
                # 为了保险，直接用索引命名，或者根据 train_vlm 判断
                name = "lr_rdt" if i == 0 else "lr_vlm"
                logs[name] = pg["lr"]
            # === 修改结束 ===

            if args.enable_distill:
                logs.update(rdt.get_loss_dict())
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # 保存：wrapper 情况下从 wrapper 里取 rdt / vlm
        if args.train_vlm and using_ds and train_model is not None:
            inner = accelerator.unwrap_model(train_model)
            inner.rdt.save_pretrained(args.output_dir)
            if args.train_vlm:
                vlm_dir = os.path.join(args.output_dir, "vlm")
                os.makedirs(vlm_dir, exist_ok=True)
                inner.vlm.save_pretrained(vlm_dir)
                processor.save_pretrained(vlm_dir)
        else:
            accelerator.unwrap_model(rdt).save_pretrained(args.output_dir)
            if args.train_vlm:
                vlm_dir = os.path.join(args.output_dir, "vlm")
                os.makedirs(vlm_dir, exist_ok=True)
                accelerator.unwrap_model(vision_language_model).save_pretrained(vlm_dir)
                processor.save_pretrained(vlm_dir)

        logger.info(f"Saved Model to {args.output_dir}")


        if args.push_to_hub:
            save_model_card(
                repo_id,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                token=args.hub_token,
                allow_patterns=["pytorch_model.bin", "*.json", "*.md"],
                # ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()
