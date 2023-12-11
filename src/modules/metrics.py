import torch.nn as nn
import torch


def dice_loss(output, target, smooth=1., predict=False):
    if predict:
        intersection = (output * target).sum(dim=1).sum(dim=1)

        loss = (1 - ((2. * intersection + smooth) / (
                output.sum(dim=1).sum(dim=1) + target.sum(dim=1).sum(dim=1) + smooth)))
        return loss.mean()
    else:
        intersection = (output * target).sum(dim=2).sum(dim=2)

        loss = (1 - ((2. * intersection + smooth) / (
                output.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def calculate_loss(pred, target, weight_contour=1.0):
    mask_target = target[:, 0, :, :].unsqueeze(1)
    contour_target = target[:, 1, :, :].unsqueeze(1)

    loss_function = nn.BCEWithLogitsLoss()

    bce_mask = loss_function(pred[:, 0, :, :].unsqueeze(1), mask_target)
    bce_contour = loss_function(pred[:, 1, :, :].unsqueeze(1), contour_target)

    pred_sigmoid = torch.sigmoid(pred)
    dice_mask = dice_loss(pred_sigmoid[:, 0, :, :].unsqueeze(1), mask_target)
    dice_contour = dice_loss(pred_sigmoid[:, 1, :, :].unsqueeze(1), contour_target)

    total_loss = bce_mask + dice_mask + weight_contour * (bce_contour + dice_contour)
    # total_loss = bce_mask + weight_contour * bce_contour
    # total_loss = bce_mask + weight_contour*bce_contour
    bce = (bce_mask, bce_contour)
    dice = (dice_mask, dice_contour)
    return total_loss, dice, bce