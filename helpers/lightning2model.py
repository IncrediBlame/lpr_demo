from DefectsModel import *

from helpers.lpr_utils import *


# Read lightning model from save
file_name = "defects_lightning_Unet_se_resnext50_32x4d_c=2_aug=False_FocalLoss_iou-0.9860.ckpt"
model_name, encoder_name, augment, out_classes, loss = decode_file_name(file_name)
best_model = DefectsModel.load_from_checkpoint(
    f"/home/incrediblame/pyscripts/defects/weights/lightning/{file_name}",
    arch=model_name, encoder_name=encoder_name, in_channels=3, out_classes=out_classes, loss=loss)
best_model = best_model.model


# Save as pytorch model
save_name = "defects_pytorch" + file_name[17: -4] + "pt"
torch.save(best_model, f"./weights/{save_name}")
