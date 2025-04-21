from pathlib import Path
import sys

THIS_FILE = Path(__file__)
# Gaussian splatting dependencies
print(str(THIS_FILE.parents[1] / "feature_3dgs"))
sys.path.append(str(THIS_FILE.parents[1] / "feature_3dgs"))

import argparse
import os
import random
import time

import numpy as np
import taichi as ti
import torch
import warp as wp
from omegaconf import OmegaConf
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from src.mpm_core import MPMModel, set_boundary_conditions
from src.utils.camera_view_utils import load_camera_params
from src.utils.misc_utils import *
from src.utils.render_utils import *


def init_training(cfg, args=None):
    # get export folder
    export_path = cfg.train.export_path if cfg.train.export_path else "./outputs"
    if cfg.train.train_tag is None:
        cfg.train.train_tag = time.strftime("%Y%m%d_%H_%M_%S")
    export_path = os.path.join(export_path, cfg.train.train_tag)
    if os.path.exists(export_path):
        if args is not None and not args.overwrite:
            overwrite = input(f"Warning: export path {export_path} already exists. Exit?(y/n)")
            if overwrite.lower() == "y":
                exit()
    else:
        os.makedirs(export_path)
        os.makedirs(os.path.join(export_path, "images"))
        os.makedirs(os.path.join(export_path, "videos"))
        os.makedirs(os.path.join(export_path, "checkpoints"))

    # set seed
    seed = cfg.train.seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # init warp
    device = f"cuda:{cfg.train.gpu}"
    wp.init()
    wp.ScopedTimer.enabled = False
    wp.set_module_options({"fast_math": False})

    # init taichi
    if cfg.preprocessing.particle_filling is not None:
        ti.init(arch=ti.cuda, device_memory_GB=8.0)

    # init torch
    torch_device = torch.device(device)
    torch.cuda.set_device(cfg.train.gpu)
    torch.backends.cudnn.benchmark = False
    print(f"\nusing device: {device}\n")

    # export config
    print(f"exporting to: {export_path}\n")
    with open(os.path.join(export_path, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    # init writer
    writer_path = os.path.join(export_path, "writers", "writer_" + time.strftime("%Y%m%d_%H_%M_%S"))
    os.makedirs(writer_path)
    writer = SummaryWriter(writer_path)

    return torch_device, export_path, writer


@torch.inference_mode()
def main(cfg, args=None):
    train_params = cfg.train
    preprocessing_params = cfg.preprocessing
    render_params = cfg.render
    material_params = cfg.material
    model_params = cfg.model
    sim_params = cfg.sim
    prompt_params = cfg.prompt_processor
    prompt_params.prompt = train_params.prompt

    # init training
    torch_device, export_path, writer = init_training(cfg, args)

    # init gaussians
    print("Initializing gaussian scene and pre-processing...")
    model_path = train_params.model_path
    gaussians = load_gaussian_ckpt(model_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device=torch_device)
        if render_params.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device=torch_device)
    )
    (mpm_params, init_e_cat, init_p_cat, unselected_params, translate_params, screen_points) = load_params(
        gaussians, pipeline, preprocessing_params, material_params, model_params, export_path=export_path
    )

    # get preprocessed gaussian params
    trans_pos = mpm_params["pos"]
    trans_cov = mpm_params["cov"]
    trans_opacity = mpm_params["opacity"]
    trans_shs = mpm_params["shs"]

    # get translation params
    rotation_matrices = translate_params["rotation_matrices"]
    scale_origin = translate_params["scale_origin"]
    original_mean_pos = translate_params["original_mean_pos"]

    gs_num = trans_pos.shape[0]

    print(f"Built gaussian particle number: {gs_num}\n")

    # camera setting
    (
        viewpoint_center_worldspace,
        observant_coordinates,
    ) = load_camera_params(render_params, rotation_matrices, scale_origin, original_mean_pos)

    # export static gaussian rendering
    print(f"Exporting static gaussian rendering to {export_path}/static.png\n")
    export_static_gaussian_rendering(
        trans_pos,
        trans_cov,
        trans_shs,
        trans_opacity,
        unselected_params,
        rotation_matrices,
        scale_origin,
        original_mean_pos,
        model_path,
        pipeline,
        render_params,
        viewpoint_center_worldspace,
        observant_coordinates,
        gaussians,
        background,
        screen_points,
        export_path,
    )

    # init mpm model and material
    print("Building MPM simulator and setting boundary conditions\n")
    mpm_model = MPMModel(sim_params, material_params, init_pos=trans_pos, enable_train=False, device=torch_device)
    set_boundary_conditions(mpm_model, sim_params.boundary_conditions)

    # material
    print("Loading neural constitutive model\n")
    # Each takes
    # - force tensor (n, 3, 3)
    # - a distribution of physicals (n, len(elasticity_physicals)), respectively, (n, len(plasticity_physicals))
    # per Gaussian particle as input
    elasticity, plasticity = init_constitute(
        material_params.elasticity,
        material_params.plasticity,
        elasticity_physicals=material_params.elasticity_physicals,
        plasticity_physicals=material_params.plasticity_physicals,
        requires_grad=False,
        device=torch_device,
    )

    num_skip_frames = sim_params.num_skip_frames
    num_frames = sim_params.num_frames
    frames_per_stage = sim_params.frames_per_stage
    assert (num_frames - num_skip_frames) % frames_per_stage == 0
    steps_per_frame = sim_params.steps_per_frame

    # init params
    x = trans_pos.detach()
    v = torch.stack(
        [torch.tensor([0.0, 0.0, -0.3], device=torch_device) for _ in range(gs_num)]
    )  # a default vertical velocity is set
    C = torch.zeros((gs_num, 3, 3), device=torch_device)
    F = torch.eye(3, device=torch_device).unsqueeze(0).repeat(gs_num, 1, 1)

    x = x.requires_grad_(False)
    v = v.requires_grad_(False)
    C = C.requires_grad_(False)
    F = F.requires_grad_(False)

    # Init physicals distribution
    # TODO: Set from our VLM prediction for each Gaussian particle
    e_cat = torch.zeros((gs_num, len(material_params.elasticity_physicals)), device=torch_device)
    p_cat = torch.zeros((gs_num, len(material_params.plasticity_physicals)), device=torch_device)

    # skip first few frames to accelerate training
    # this frames are meaningless when there is no contact or collision
    mpm_model.reset()
    for frame in tqdm(range(num_skip_frames + num_frames), desc="Frame"):
        # render
        frame_id = frame
        # get rendering params
        (render_pos, render_cov, render_shs, render_opacity, render_rot) = get_mpm_gaussian_params(
            pos=x,
            cov=trans_cov,
            shs=trans_shs,
            opacity=trans_opacity,
            F=F,
            unselected_params=unselected_params,
            rotation_matrices=rotation_matrices,
            scale_origin=scale_origin,
            original_mean_pos=original_mean_pos,
        )

        rendering = render_mpm_gaussian(
            model_path=model_path,
            pipeline=pipeline,
            render_params=render_params,
            step=frame_id,
            viewpoint_center_worldspace=viewpoint_center_worldspace,
            observant_coordinates=observant_coordinates,
            gaussians=gaussians,
            background=background,
            pos=render_pos,
            cov=render_cov,
            shs=render_shs,
            opacity=render_opacity,
            rot=render_rot,
            screen_points=screen_points,
        )
        if train_params.export_video:
            export_rendering(rendering, frame_id, folder=os.path.join(export_path, "images"))

        for step in range(steps_per_frame):
            # mpm step, using checkpoint to save memory
            stress = checkpoint(elasticity, F, e_cat)

            assert torch.all(torch.isfinite(stress))

            x, v, C, F = checkpoint(mpm_model, x, v, C, F, stress)

            assert torch.all(torch.isfinite(x))
            assert torch.all(torch.isfinite(F))

            F = checkpoint(plasticity, F, p_cat)

            assert torch.all(torch.isfinite(F))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")

    parser.add_argument(
        "--guidance_config",
        type=str,
        default="./configs/ms_guidance.yaml",
        help="Path to the SDS guidance config file.",
    )

    parser.add_argument("--test", action="store_true", help="Test mode.")

    parser.add_argument("--gpu", type=int, help="GPU index.")

    parser.add_argument("--tag", type=str, help="Training tag.")

    parser.add_argument("--overwrite", "-o", action="store_true", help="Overwrite the existing export folder.")

    parser.add_argument("--output", type=str, help="Output folder.")

    parser.add_argument("--save_internal", action="store_true", help="Save internal checkpoints.")

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    guidance_cfg = OmegaConf.load(args.guidance_config)
    cfg = OmegaConf.merge(cfg, guidance_cfg)

    if args.gpu is not None:
        cfg.train.gpu = args.gpu
    if args.test:
        cfg.train.enable_train = False
    if args.tag is not None:
        cfg.train.train_tag = args.tag
    if args.output is not None:
        cfg.train.export_path = args.output

    main(cfg, args)
