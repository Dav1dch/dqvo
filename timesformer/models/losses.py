# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]


def quaternion_loss(pred_quat, target_quat):
    """
    Compute quaternion loss between predicted and target quaternions.
    
    Args:
        pred_quat: predicted quaternion [qw, qx, qy, qz] (batch_size, 4)
        target_quat: target quaternion [qw, qx, qy, qz] (batch_size, 4)
        
    Returns:
        loss: quaternion loss (scalar)
    """
    # Ensure unit quaternions
    pred_quat = pred_quat / torch.norm(pred_quat, dim=-1, keepdim=True)
    target_quat = target_quat / torch.norm(target_quat, dim=-1, keepdim=True)
    
    # Compute dot product
    dot = torch.sum(pred_quat * target_quat, dim=-1)
    
    # Ensure dot product is in [-1, 1]
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # Compute angular distance
    angle = 2 * torch.acos(torch.abs(dot))
    
    # Return squared angular distance
    return torch.mean(angle ** 2)


def dual_quaternion_loss(pred_dq, target_dq):
    """
    Compute dual quaternion loss between predicted and target dual quaternions.
    
    Args:
        pred_dq: predicted dual quaternion [qw, qx, qy, qz, tx, ty, tz] (batch_size, 7)
        target_dq: target dual quaternion [qw, qx, qy, qz, tx, ty, tz] (batch_size, 7)
        
    Returns:
        loss: dual quaternion loss (scalar)
    """
    # Extract rotation quaternions and translations
    pred_q = pred_dq[:, :4]
    target_q = target_dq[:, :4]
    pred_t = pred_dq[:, 4:]
    target_t = target_dq[:, 4:]
    
    # Compute quaternion loss for rotation
    rot_loss = quaternion_loss(pred_q, target_q)
    
    # Compute translation loss (MSE)
    trans_loss = torch.mean((pred_t - target_t) ** 2)
    
    # Combine losses
    total_loss = rot_loss + trans_loss
    
    return total_loss


def quaternion_loss_weighted(pred_dq, target_dq, weight_rot=1.0, weight_trans=1.0):
    """
    Compute weighted dual quaternion loss.
    
    Args:
        pred_dq: predicted dual quaternion [qw, qx, qy, qz, tx, ty, tz] (batch_size, 7)
        target_dq: target dual quaternion [qw, qx, qy, qz, tx, ty, tz] (batch_size, 7)
        weight_rot: weight for rotation loss
        weight_trans: weight for translation loss
        
    Returns:
        loss: weighted dual quaternion loss (scalar)
    """
    # Extract rotation quaternions and translations
    pred_q = pred_dq[:, :4]
    target_q = target_dq[:, :4]
    pred_t = pred_dq[:, 4:]
    target_t = target_dq[:, 4:]
    
    # Compute quaternion loss for rotation
    rot_loss = quaternion_loss(pred_q, target_q)
    
    # Compute translation loss (MSE)
    trans_loss = torch.mean((pred_t - target_t) ** 2)
    
    # Combine losses with weights
    total_loss = weight_rot * rot_loss + weight_trans * trans_loss
    
    return total_loss
