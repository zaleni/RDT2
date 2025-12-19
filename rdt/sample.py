from collections import defaultdict
import torch
import torch.nn.functional as F
from data.umi.pose_util import geodesic_loss, rot6d_to_mat_torch

def compute_action_errors(
    accelerator,
    pred_action, gt_action,
    num_robot,
    action_dim_per_robot: int, # 去掉默认值，强制传入
):
    B, T, D = pred_action.shape
    action_shape = int(D / num_robot)
    
    # 1. 验证维度
    assert action_shape == action_dim_per_robot, (
        f"Unexpected action dim per robot. "
        f"Got total_dim={D}, num_robot={num_robot}, per_robot={action_shape}, "
        f"expected_per_robot={action_dim_per_robot}"
    )

    pred_action = pred_action.view(B, T, -1, action_shape)
    gt_action = gt_action.view(B, T, -1, action_shape)

    result = {}
    
    # 2. 根据维度区分计算逻辑
    # === 情况 A: 标准 UMI EEF 空间 (10 dim = 3 pos + 6 rot + 1 width) ===
    if action_dim_per_robot == 10:
        # Use geodesic loss for rotation
        pred_rot6d = pred_action[..., 3:9]
        gt_rot6d = gt_action[..., 3:9]
        
        # 转换并计算旋转误差
        pred_rot_mat = rot6d_to_mat_torch(pred_rot6d).to(dtype=gt_rot6d.dtype)
        gt_rot_mat = rot6d_to_mat_torch(gt_rot6d)
        rot_error = geodesic_loss(pred_rot_mat, gt_rot_mat, reduce=True, return_degrees=True)

        result['action_mse_error'] = F.mse_loss(pred_action, gt_action)
        result['action_mse_error_pos'] = F.mse_loss(pred_action[..., :3], gt_action[..., :3])
        result['action_geodesic_error_rot'] = rot_error
        result['action_mse_error_width'] = F.mse_loss(pred_action[..., 9], gt_action[..., 9])

    # === 情况 B: 关节空间 Joint Space (7 dim = 6 joints + 1 gripper or 7 joints) ===
    elif action_dim_per_robot == 7:
        # 关节空间直接计算整体 MSE
        result['action_mse_error'] = F.mse_loss(pred_action, gt_action)
        # 可选：如果您想看前6个关节和夹爪的区别，可以加细分
        # result['action_mse_joints'] = F.mse_loss(pred_action[..., :6], gt_action[..., :6])
        # result['action_mse_gripper'] = F.mse_loss(pred_action[..., 6], gt_action[..., 6])

    # === 其他情况 ===
    else:
        # 默认只算 MSE
        result['action_mse_error'] = F.mse_loss(pred_action, gt_action)

    gathered_result = {}
    for k, v in result.items():
        gathered_result[k] = accelerator.gather(v).mean().item()

    return gathered_result

@torch.no_grad()
def log_sample_res(
    vision_language_model, selected_layers,
    vision_encoder, rdt, normalizer, args,
    accelerator, weight_dtype, dataloader, logger
):
    logger.info(
        f"Running sampling for {args.num_sample_batches} batches..."
    )

    rdt.eval()

    loss_for_log = defaultdict(float)
    loss_counter = defaultdict(int)
    for step, batch in enumerate(dataloader):
        if step >= args.num_sample_batches:
            break

        # Keep device consistent with model
        actions = batch["actions"].to(device=rdt.device, dtype=weight_dtype)
        states = batch["states"].to(device=rdt.device, dtype=weight_dtype)

        if vision_encoder is not None:
            images = {k: v.to(device=rdt.device, dtype=weight_dtype) for k, v in batch["images"].items()}
            k = next(iter(images))
            batch_size, _, C, H, W = images[k].shape
            for k in images:
                images[k] = images[k].reshape(-1, C, H, W)
            image_embeds = vision_encoder(images).detach()
            image_embeds = image_embeds.reshape((batch_size, -1, vision_encoder.embed_dim))
        else:
            image_embeds = None

        vlang_attn_mask = batch["vision_language_model_inputs"]["attention_mask"].to(device=rdt.device, dtype=torch.bool)
        outputs = vision_language_model(
            **{k: v.to(device=rdt.device) if hasattr(v, "to") else v for k, v in batch["vision_language_model_inputs"].items()},
            use_cache=True,
        )

        if isinstance(selected_layers, list):
            vlang_kv_cache = [outputs.past_key_values[i] for i in selected_layers]
        else:
            vlang_kv_cache = [outputs.past_key_values[selected_layers]]

        pred_nsamples = rdt.predict_action(
            lang_kv_cache=vlang_kv_cache,
            lang_attn_mask=vlang_attn_mask,
            img_tokens=image_embeds,
            state_tokens=states,
        )
        pred_actions = normalizer["action"].unnormalize(pred_nsamples).to(rdt.device)

        # === [关键修改] 自动推断 num_robot 和 action_dim_per_robot ===
        total_dim = actions.shape[-1]
        
        # 假设双臂配置 (num_robot=2)
        # 14 维 -> 每臂 7 维 (您的关节数据)
        # 20 维 -> 每臂 10 维 (标准 EEF 数据)
        if total_dim == 14:
            action_dim_per_robot = 7
            num_robot = 2
        elif total_dim == 20:
            action_dim_per_robot = 10
            num_robot = 2
        elif total_dim == 7: # 单臂关节
            action_dim_per_robot = 7
            num_robot = 1
        elif total_dim == 10: # 单臂 EEF
            action_dim_per_robot = 10
            num_robot = 1
        else:
            # Fallback (如果不确定，默认设为 total_dim，当做单机器人)
            action_dim_per_robot = total_dim
            num_robot = 1

        # 传递推断出的参数
        action_errors = compute_action_errors(
            accelerator, pred_actions, actions, num_robot=num_robot, action_dim_per_robot=action_dim_per_robot
        )
        
        for k, v in action_errors.items():
            loss_for_log[k] += v
            loss_counter[k] += 1

    for name in loss_for_log:
        loss_for_log[name] = loss_for_log[name] / loss_counter[name]

    rdt.train()
    torch.cuda.empty_cache()

    return dict(loss_for_log)