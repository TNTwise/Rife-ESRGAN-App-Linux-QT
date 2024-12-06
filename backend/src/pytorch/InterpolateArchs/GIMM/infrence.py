from gimmvfi_r import GIMMVFI_R
import torch
import os
from PIL import Image
import numpy as np



device = torch.device("cuda")
model = GIMMVFI_R().to(device)





ckpt = torch.load("models/GIMM-VFI/gimmvfi_r_arb_lpips.pt", map_location="cpu")
model.load_state_dict(ckpt["state_dict"], strict=True)

def load_image(img_path):
    img = Image.open(img_path)
    raw_img = np.array(img.convert("RGB"))
    img = torch.from_numpy(raw_img.copy()).permute(2, 0, 1) / 255.0
    return img.to(torch.float).unsqueeze(0)


img_path0 = os.path.join(source_path, img_list[j])
        img_path2 = os.path.join(source_path, img_list[j + 1])
        # prepare data b,c,h,w
        I0 = load_image(img_path0)
        if j == start:
            images.append(
                (I0.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255.0)[
                    :, :, ::-1
                ].astype(np.uint8)
            )
            ori_image.append(
                (I0.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255.0)[
                    :, :, ::-1
                ].astype(np.uint8)
            )
            images[-1] = cv2.hconcat([ori_image[-1], images[-1]])
        I2 = load_image(img_path2)
        padder = InputPadder(I0.shape, 32)
        I0, I2 = padder.pad(I0, I2)
        xs = torch.cat((I0.unsqueeze(2), I2.unsqueeze(2)), dim=2).to(
            device, non_blocking=True
        )
        model.eval()
        batch_size = xs.shape[0]
        s_shape = xs.shape[-2:]

        model.zero_grad()
        ds_factor = args.ds_factor
        with torch.no_grad():
            coord_inputs = [
                (
                    model.sample_coord_input(
                        batch_size,
                        s_shape,
                        [1 / args.N * i],
                        device=xs.device,
                        upsample_ratio=ds_factor,
                    ),
                    None,
                )
                for i in range(1, args.N)
            ]
            timesteps = [
                i * 1 / args.N * torch.ones(xs.shape[0]).to(xs.device).to(torch.float)
                for i in range(1, args.N)
            ]
            all_outputs = model(xs, coord_inputs, t=timesteps, ds_factor=ds_factor)
            out_frames = [padder.unpad(im) for im in all_outputs["imgt_pred"]]
            out_flowts = [padder.unpad(f) for f in all_outputs["flowt"]]
        flowt_imgs = [
            flow_to_image(
                flowt.squeeze().detach().cpu().permute(1, 2, 0).numpy(),
                convert_to_bgr=True,
            )
            for flowt in out_flowts
        ]
        I1_pred_img = [
            (I1_pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0)[
                :, :, ::-1
            ].astype(np.uint8)
            for I1_pred in out_frames
        ]

        for i in range(args.N - 1):
            images.append(I1_pred_img[i])
            flows.append(flowt_imgs[i])

            images[-1] = cv2.hconcat([ori_image[-1], images[-1]])

        images.append(
            (
                (padder.unpad(I2)).squeeze().detach().cpu().numpy().transpose(1, 2, 0)
                * 255.0
            )[:, :, ::-1].astype(np.uint8)
        )
        ori_image.append(
            (
                (padder.unpad(I2)).squeeze().detach().cpu().numpy().transpose(1, 2, 0)
                * 255.0
            )[:, :, ::-1].astype(np.uint8)
        )
        images[-1] = cv2.hconcat([ori_image[-1], images[-1]])
