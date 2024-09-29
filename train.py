import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monai.data import decollate_batch
from monai.transforms import SaveImage
from pytorch3d.structures import Meshes

from utils import get_each_mesh


class Trainer:
    def __init__(
            self,
            model,
            optimizer,
            loss_function,
            train_loader,
            val_loader,
            saler,
            logger,
            eval_num,
            max_epoches,
            dataset_name,
            model_name,
            post_label,
            post_pred,
            dice_metric,
            filename,
            roi_size,
            save_pred=False,
            save_dice_csv=False,
            num_classes=31,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scaler = saler
        self.logger = logger
        self.eval_num = eval_num
        self.max_epoches = max_epoches
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.save_dice_csv = save_dice_csv
        self.post_label = post_label
        self.post_pred = post_pred
        self.dice_metric = dice_metric
        self.save_pred = save_pred
        self.filename = filename
        self.num_classes = num_classes
        self.roi_size = roi_size

        self.epoch_loss_values = []
        self.metric_values = []

    # # label to mesh
    @staticmethod
    def labelmesh(label):
        all_cls = torch.unique(label)
        meshes = {}
        for cls in all_cls[1:]:
            mesh = get_each_mesh(label.cpu(), cls)
            meshes[cls] = Meshes(verts=[mesh.vertices], faces=[mesh.faces])

        return meshes

    # section epoch train
    def train(self, global_epoch, train_loader, dice_val_best, global_step_best):
        self.model.train()
        epoch_loss = 0
        step = 0
        for step, batch in enumerate(train_loader):
            step += 1
            device = next(self.model.parameters()).device
            x, y = (batch["image"].to(device), batch["label"].to(device))
            self.optimizer.zero_grad()

            logit_map, iter_meshes, vertices = self.model(x, label=y)
            loss = self.loss_function((logit_map, iter_meshes, vertices), y)

            epoch_loss += loss.item()
            self.logger.info(
                f"Training {global_epoch}: ({step} / {len(train_loader)} Steps) (loss={loss:.3f})"
            )
            if device == torch.device("cuda"):
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

        epoch_loss /= step
        self.epoch_loss_values.append(epoch_loss)
        self.logger.info(f"Epoch {global_epoch} average loss: {epoch_loss:.4f}")

        if (
                global_epoch % self.eval_num == 0 and global_epoch != 0
        ) or global_epoch == self.max_epoches:
            dice_val = self.validation(self.val_loader)
            self.metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_epoch
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        f"./{self.model_name}/{self.dataset_name}/",
                        f"best_metric_model_{dice_val}.pth",
                    ),
                )
                self.logger.info(
                    f"Model Was Saved ! Current Best Avg. Dice: {dice_val_best: .3f} Current Avg. Dice: {dice_val: .3f}"
                )
            else:
                self.logger.info(
                    f"Model Was Not Saved ! Current Best Avg. Dice: {dice_val_best: .3f} Current Avg. Dice: {dice_val: .3f}"
                )

            if not self.save_dice_csv or not self.save_pred:
                plt.figure("train", (12, 6))
                plt.subplot(1, 2, 1)
                plt.title("Iteration Average Loss")
                x = [i + 1 for i in range(len(self.epoch_loss_values))]
                y = self.epoch_loss_values
                plt.xlabel("Iteration")
                plt.plot(x, y)
                plt.subplot(1, 2, 2)
                plt.title("Val Mean Dice")
                x = [self.eval_num * (i + 1) for i in range(len(self.metric_values))]
                y = self.metric_values
                plt.xlabel("Iteration")
                plt.plot(x, y)
                plt.savefig(f"./{self.model_name}/{self.dataset_name}/loss.png")

        global_epoch += 1
        return global_epoch, dice_val_best, global_step_best

    def validation(self, epoch_iterator_val):
        self.model.eval()
        with torch.no_grad():
            # dice_list = []
            if self.save_dice_csv:
                cols = ["cls" + str(i) for i in range(self.num_classes)]
                cols += ["mean_dice"]
                df = pd.DataFrame(columns=cols)

            for i, batch in enumerate(epoch_iterator_val):
                device = next(self.model.parameters()).device
                val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
                with torch.cuda.amp.autocast():
                    val_outputs = self.model(val_inputs)
                    batch["pred"] = val_outputs
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [
                    self.post_label(val_label_tensor)
                    for val_label_tensor in val_labels_list
                ]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [
                    self.post_pred(val_pred_tensor)
                    for val_pred_tensor in val_outputs_list
                ]
                current_dice = self.dice_metric(
                    y_pred=val_output_convert, y=val_labels_convert
                )
                self.logger.info(
                    f"Validate {i + 1} / {len(epoch_iterator_val)}: dice={current_dice.nanmean().item():.4f}"
                )
                if self.save_dice_csv:
                    current_dice = current_dice.cpu().numpy()
                    df.loc[i, "cls0":f"cls{self.num_classes - 1}"] = current_dice
                    df.loc[i, "mean_dice"] = np.nanmean(current_dice)

                if self.save_pred:
                    # save pred
                    val_pred = torch.argmax(
                        val_outputs_list[0], dim=0, keepdim=True
                    )  # (1, h, w, d)
                    SaveImage(
                        output_dir=self.filename,
                        output_postfix=f"few_shot_pred{i}",
                        separate_folder=False,
                        resample=False,
                    )(
                        val_pred,
                    )

            if self.save_dice_csv:
                os.makedirs(os.path.dirname(self.filename), exist_ok=True)
                with pd.ExcelWriter(self.filename, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name=f"{self.model_name}", index=False)

            mean_dice_val = self.dice_metric.aggregate().item()
            self.dice_metric.reset()
        return mean_dice_val
