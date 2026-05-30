import torch
import torch.nn as nn
import torch.nn.functional as F

from util.misc import NestedTensor

class DinoBloom(nn.Module):
    def __init__(self, 
                model_path = "/home/iml/DINO_X/DinoBloom-S.pth", 
                model_name = "dinov2_vits14",
                patch_size=14,
            ):
        super().__init__()
        
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)

        pretrained = torch.load(model_path, map_location=torch.device("cpu"))

        new_state_dict = {}
        for k, v in pretrained["teacher"].items():
            if "dino_head" in k or "ibot_head" in k:
                continue
            else:
                new_k = k.replace('backbone.', '')
                new_state_dict[new_k] = v

        self.model.pos_embed = nn.Parameter(torch.zeros(1, 257, 384))

        self.model.load_state_dict(new_state_dict, strict=True)
        self.patch_size = patch_size
        
    @torch.no_grad()
    def forward(self, x):
        _, _, h, w = x.tensors.shape
        new_h = (h + 13) // 14 * 14  
        new_w = (w + 13) // 14 * 14

        if new_h != h or new_w != w:
            x.tensors = F.interpolate(x.tensors, size=(new_h, new_w), mode='bilinear', align_corners=False)
        mask = x.mask
        # x_cls = self.model.forward_features(x.tensors)["x_norm_clstoken"]
        # x = self.model.forward_features(x.tensors)["x_norm_patchtokens"]
        features = self.model.forward_features(x.tensors)
        x_cls = features["x_norm_clstoken"]
        x = features["x_norm_patchtokens"]
        bs, num_patches, embed_dim = x.shape
        H = int(new_h / self.patch_size)
        W = int(new_w / self.patch_size)
        assert H * W == num_patches, f"Mismatch: {H} * {W} != {num_patches}"

        x = x.permute(0, 2, 1).reshape(bs, embed_dim, H, W)
        x_mask = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]

        x_upsampled_0 = F.interpolate(x, scale_factor=2)
        x_upsampled_0_mask = F.interpolate(mask[None].float(), size=x_upsampled_0.shape[-2:]).to(torch.bool)[0]

        x_downsampled_1= F.interpolate(x, scale_factor=(1/2))
        x_downsampled_1_mask = F.interpolate(mask[None].float(), size=x_downsampled_1.shape[-2:]).to(torch.bool)[0]

        return [NestedTensor(x_upsampled_0, x_upsampled_0_mask), NestedTensor(x, x_mask), NestedTensor(x_downsampled_1, x_downsampled_1_mask)] , x_cls




if __name__ == "__main__":
    x = torch.randn((1, 3, 224, 224))
    model = DinoBloom()

    out = model(x)
    print(out[0].shape, out[1].shape, out[2].shape)

