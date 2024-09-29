from calendar import c
import os
import torch
from monai.transforms import (
    AsDiscrete,
)
from monai.config import print_config
from monai.metrics import DiceMetric

from logger import setup_logger
from utils import load_pretrained
from data.dataloader import get_loader
from train import Trainer
from meshfit import MeshFit
from loss import Loss

# ! section: parameters
# dataset_name = "SegRap"
dataset_name = "HaN"
model_name = "UNet"

max_epochs = 2000
eval_num = 5
global_epoch = 1  # record the current epoch
dice_val_best = 0.0
global_epoch_best = 0
log_name = "train"

save_dice_csv = False
save_pred = False
save_pred_index = None
current_dir = os.path.dirname(__file__)

# The address of the pre-trained model for the entire model
pretrained_dir = None

ce_weight = None
in_channels = 1
num_classes = 10
roi_size = (96, 96, 96)

filename = None
if save_dice_csv:
    eval_num = 1
    filename = f"./{model_name}/{dataset_name}/{model_name}_dice.xlsx"
    log_name = "dice_csv"
    max_epochs = 2
if save_pred:
    eval_num = 1
    log_name = "pred"
    max_epochs = 2
    filename = f"./{model_name}/{dataset_name}/"

print_config()
logger = setup_logger(
    output=os.path.join(current_dir, f"{model_name}/{dataset_name}/"),
    distributed_rank=0,
    name=log_name,
)

# ! section: load the model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

model = MeshFit(
    model_name,
    in_channels,
    32,
    num_classes,
    roi_size,
)


pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Total parameters count: {pytorch_total_params}")

# ! Load pretrained model weights from pretrained_dir
if pretrained_dir:
    try:
        load_pretrained(model, pretrained_dir, logger, strict=False)
        # load_pretrained(pretrained_model, pretrained_dir, logger, strict=False)
    except Exception as e:
        logger.error(
            "Failed to load pretrained model weights from {} due to {}".format(
                pretrained_dir, e
            )
        )
        raise e
    else:
        logger.info(
            "Successfully loaded pretrained model weights from {}".format(
                pretrained_dir
            )
        )

# ! section: load the data
train_loader, val_loader = get_loader(
    dataset_name, save_dice_csv, save_pred, save_pred_index
)

model = model.to(device)

# section optimizer and loss
torch.backends.cudnn.benchmark = True
loss_function = Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler()
post_label = AsDiscrete(to_onehot=num_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

# section training
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_function=loss_function,
    train_loader=train_loader,
    val_loader=val_loader,
    saler=scaler,
    logger=logger,
    eval_num=eval_num,
    max_epoches=max_epochs,
    dataset_name=dataset_name,
    model_name=model_name,
    post_label=post_label,
    post_pred=post_pred,
    dice_metric=dice_metric,
    roi_size=roi_size,
    filename=filename,
    save_pred=save_pred,
    save_dice_csv=save_dice_csv,
    num_classes=num_classes,
)

while global_epoch < max_epochs:
    global_epoch, dice_val_best, global_epoch_best = trainer.train(
        global_epoch, train_loader, dice_val_best, global_epoch_best
    )

logger.info(
    f"train completed, best_metric: {dice_val_best:.4f} "
    f"at iteration: {global_epoch_best}"
)
