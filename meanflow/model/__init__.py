from .unet import UNet

def create_model(args,):
    if args.use_edm_aug:
        augment_dim = 6
    else:
        augment_dim = 0

    if args.class_cond:
        label_dim = args.num_classes
    else:
        label_dim = 0

    if args.model == "unet":
        model = UNet(
            img_resolution = args.img_size,
            in_channels    = args.img_dim,
            out_channels   = args.img_dim,
            label_dim      = label_dim,
            model_channels = 64,
            channel_mult   = [1, 2, 3, 4],
            channel_mult_emb = 4,
            channel_mult_noise = 2,
            num_blocks       = 4,
            attn_resolutions = [16],
            augment_dim      = augment_dim,
            dropout          = args.dropout,
            label_dropout    = 0.0,
        )

    else:
        raise NotImplementedError(" > unknown model name: ", args.model)
    
    return model