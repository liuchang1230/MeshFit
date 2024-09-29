from monai.networks.nets import SwinUNETR, UNETR, VNet, SegResNet
from UNet_model import UNet


def get_model(model_name, in_channels=1, out_channels=14, roi_size=(96, 96, 96)):
    if model_name == "UNet":
        model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2),
        )
    elif model_name == "VNet":
        model = VNet(
            in_channels=in_channels,
            out_channels=out_channels,
        )
    elif model_name == "SegResNet":
        model = SegResNet(
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=32,
            blocks_down=(1, 2, 2, 2, 2),
            blocks_up=(1, 1, 1, 1),
        )
    elif model_name == "UNETR":
        model = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=roi_size,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )
    elif model_name == "SwinUNETR":
        model = SwinUNETR(
            img_size=roi_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=48,
            use_checkpoint=True,
        )
    else:
        raise ValueError(f"model: {model_name} is not supported yet.")

    return model
