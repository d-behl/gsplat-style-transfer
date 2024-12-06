import argparse
import math
import os
import os.path as osp
from pyexpat import features
import time
from typing import Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import viser
from gsplat._helper import load_test_data
from gsplat.rendering import rasterization

import nerfview

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir", type=str, default="results/", help="where to dump outputs"
)
parser.add_argument(
    "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
)
parser.add_argument(
    "--ckpt", type=str, default="styled_figurines.pt", help="path to the .pt file"
)
parser.add_argument("--port", type=int, default=8080, help="port for the viewer server")
parser.add_argument(
    "--backend", type=str, default="gsplat", help="gsplat, gsplat_legacy, inria"
)
args = parser.parse_args()
assert args.scene_grid % 2 == 1, "scene_grid must be odd"

torch.manual_seed(42)
device = "cuda"

ckpt = torch.load(args.ckpt, map_location=device)
means = ckpt["means"]
quats = ckpt["quats"]
scales = torch.exp(ckpt["scales"])
opacities = torch.sigmoid(ckpt["opacities"]).squeeze(-1)
features_dc = ckpt["features_dc"]
colors = torch.sigmoid(features_dc).squeeze(-1)
sh_degree = None

# crop
# aabb = torch.tensor((-1.0, -1.0, -1.0, 1.0, 1.0, 0.7), device=device)
# edges = aabb[3:] - aabb[:3]
# sel = ((means >= aabb[:3]) & (means <= aabb[3:])).all(dim=-1)
# sel = torch.where(sel)[0]
# means, quats, scales, colors, opacities = (
#     means[sel],
#     quats[sel],
#     scales[sel],
#     colors[sel],
#     opacities[sel],
# )

# repeat the scene into a grid (to mimic a large-scale setting)
# repeats = args.scene_grid
# gridx, gridy = torch.meshgrid(
#     [
#         torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
#         torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
#     ],
#     indexing="ij",
# )
# grid = torch.stack([gridx, gridy, torch.zeros_like(gridx)], dim=-1).reshape(-1, 3)
# means = means[None, :, :] + grid[:, None, :] * edges[None, None, :]
# means = means.reshape(-1, 3)
# quats = quats.repeat(repeats**2, 1)
# scales = scales.repeat(repeats**2, 1)
# colors = colors.repeat(repeats**2, 1, 1)
# opacities = opacities.repeat(repeats**2)
# print("Number of Gaussians:", len(means))


# register and open viewer
@torch.no_grad()
def viewer_render_fn(camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
    width, height = img_wh
    c2w = camera_state.c2w
    K = camera_state.get_K(img_wh)
    c2w = torch.from_numpy(c2w).float().to(device)
    K = torch.from_numpy(K).float().to(device)
    viewmat = c2w.inverse()

    if args.backend == "gsplat":
        rasterization_fn = rasterization
    elif args.backend == "gsplat_legacy":
        from gsplat import rasterization_legacy_wrapper

        rasterization_fn = rasterization_legacy_wrapper
    elif args.backend == "inria":
        from gsplat import rasterization_inria_wrapper

        rasterization_fn = rasterization_inria_wrapper
    else:
        raise ValueError

    render_colors, render_alphas, meta = rasterization_fn(
        means,  # [N, 3]
        quats,  # [N, 4]
        scales,  # [N, 3]
        opacities,  # [N]
        colors,  # [N, 3]
        viewmat[None],  # [1, 4, 4]
        K[None],  # [1, 3, 3]
        width,
        height,
        sh_degree=sh_degree,
        render_mode="RGB",
        backgrounds=torch.ones(1, 3, device=device),
        # this is to speedup large-scale rendering by skipping far-away Gaussians.
        # radius_clip=3,
    )
    render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()
    return render_rgbs


server = viser.ViserServer(port=args.port, verbose=False)
_ = nerfview.Viewer(
    server=server,
    render_fn=viewer_render_fn,
    mode="rendering",
)
print("Viewer running... Ctrl+C to exit.")
time.sleep(100000)
