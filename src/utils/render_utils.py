import json
import math
import os
import logging
from typing import TypedDict

import cv2
import imageio
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn.functional as F

from .camera_view_utils import get_camera_view
from .filling_utils import *
from .transformation_utils import *

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from gaussian_renderer import GaussianModel
from scene.gaussian_model import GaussianModel
from utils.sh_utils import RGB2SH, eval_sh
from utils.system_utils import searchForMaxIteration


_logger = logging.getLogger(__name__)


class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

def load_gaussian_ckpt(model_path, sh_degree=3, iteration=-1):
    # Find checkpoint
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )

    # Load guassians
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians

def initialize_resterize(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
):
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterize = GaussianRasterizer(raster_settings=raster_settings)
    return rasterize


def load_params_from_gs(
    pc: GaussianModel, pipe, scaling_modifier=1.0, override_color=None
):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        shs = pc.get_features
    else:
        colors_precomp = override_color

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    semantic_feature = pc.get_semantic_feature
    return {
        "pos": means3D,
        "screen_points": means2D,
        "shs": shs,
        "colors_precomp": colors_precomp,
        "opacity": opacity,
        "scales": scales,
        "rotations": rotations,
        "cov3D_precomp": cov3D_precomp,
        "semantic_feature": semantic_feature,
    }


class MaterialPred(TypedDict):
    semantic_feature: list[float]
    log_E: float
    nu: float
    density: float
    elasticity_model: str
    plasticity_model: str


def load_material_preds(material_preds_path: str) -> list[MaterialPred]:
    with open(material_preds_path, "r") as f:
        material_preds = json.load(f)
    return material_preds

def create_assignment_vis_3dgs(gaussians: GaussianModel, best_match_idx: torch.Tensor, export_path: str, selected_mask: torch.BoolTensor):
    _logger.info("Creating material assignment visualization...")
    # Create a new PLY file for the assignment visualization
    gaussians.save_ply(export_path)

    colored_gaussians = GaussianModel(gaussians.active_sh_degree)
    colored_gaussians.load_ply(export_path)

    # Create a categorical colormap tensor with RGB values in [0, 1] range
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('tab10')
    cmap_colors = [cmap(i)[:3] for i in range(10)]  # Get RGB values (exclude alpha)
    cmap_tensor = torch.tensor(cmap_colors, device=best_match_idx.device, dtype=torch.float32)

    # Each Gaussian that is assgined the same material will have the same color
    colored_gaussians._features_dc[selected_mask] = RGB2SH(cmap_tensor[best_match_idx % len(cmap_tensor)])

    # Set color of unselected/not-simulated Gaussians to black
    # colored_gaussians._features_dc[~selected_mask, 0] = RGB2SH(torch.zeros_like(cmap_tensor[0]))
    colored_gaussians.save_ply(export_path)
    _logger.info("Done...")


def transfer_material_params(semantic_feature: torch.Tensor, material_preds: list[MaterialPred], material_params) -> dict[str, torch.Tensor]:
    """Transfer material properties of the closest material prediction in terms of similarity of semantic features.

    Args:
        semantic_feature: semantic features of the Gaussian
        material_params: list of material predictions

    Returns:
        Dictionary of per-Gaussian material properties.
    """
    _logger.info("Transferring material parameters...")
    device = semantic_feature.device

    # Compute similarity between query and material prediction semantic features
    query_semantic_features = torch.stack([torch.tensor(material_pred["semantic_feature"], device=device) for material_pred in material_preds])
    query_semantic_features = F.normalize(query_semantic_features, dim=1, p=2)
    normed_semantic_feature = F.normalize(semantic_feature, dim=1, p=2)
    sim_matrix = torch.matmul(normed_semantic_feature.squeeze(1), query_semantic_features.T)

    # Assign the material properties with the closest semantic feature to each Gaussian
    similarity, best_match_idx = torch.max(sim_matrix, dim=1)  # (num_gaussians, )

    # Extract material properties from the best match
    # Create lookup tensors for each property
    log_E_lookup = torch.tensor([pred["log_E"] for pred in material_preds], device=device)
    nu_lookup = torch.tensor([pred["nu"] for pred in material_preds], device=device)
    density_lookup = torch.tensor([pred["density"] for pred in material_preds], device=device)
    # Map elasticity and plasticity models to indices
    elasticity_indices = {model: idx for idx, model in enumerate(material_params.elasticity_physicals)}
    plasticity_indices = {model: idx for idx, model in enumerate(material_params.plasticity_physicals)}
    e_cat_lookup = torch.tensor([elasticity_indices[pred["elasticity_model"]] for pred in material_preds], 
                               device=device, dtype=torch.long)
    p_cat_lookup = torch.tensor([plasticity_indices[pred["plasticity_model"]] for pred in material_preds], 
                               device=device, dtype=torch.long)

    # Use lookup tables
    log_E = torch.index_select(log_E_lookup, 0, best_match_idx)
    nu = torch.index_select(nu_lookup, 0, best_match_idx)
    density = torch.index_select(density_lookup, 0, best_match_idx)
    e_cat = torch.index_select(e_cat_lookup, 0, best_match_idx)
    p_cat = torch.index_select(p_cat_lookup, 0, best_match_idx)
    _logger.info("Transferring material parameters... Done")

    out = {
        "log_E": log_E,
        "nu": nu,
        "density": density,
        "similarity": similarity,
        "best_match_idx": best_match_idx,
        "e_cat": e_cat,
        "p_cat": p_cat,
    }
    return {k: v.unsqueeze(1) for k, v in out.items()}


def load_params(gaussians: GaussianModel, pipeline: PipelineParamsNoparse, preprocessing_params: DictConfig, material_params: DictConfig, material_preds: list[MaterialPred] | None = None, export_path: str = './'):
    params = load_params_from_gs(gaussians, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]
    init_semantic_feature = params["semantic_feature"]

    # throw away low opacity kernels
    mask = init_opacity[:, 0] > preprocessing_params.opacity_threshold
    sel_gs_indices = torch.nonzero(mask).squeeze()
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]
    init_semantic_feature = init_semantic_feature[mask, :]

    # throw away large kernels
    # this is useful when the Gaussian asset is of low quality
    init_cov = init_cov * 0.5
    init_cov_mat = get_mat_from_upper(init_cov)
    mask = filter_cov(init_cov_mat, threshold=2e-4)
    init_cov[~mask] = 0.5 * init_cov[~mask]

    # rotate and translate
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params.rotation_degree),
        preprocessing_params.rotation_axis,
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)

    # select a sim area and save params of unselected particles for later rendering
    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (None, None, None, None)
    if preprocessing_params.sim_area is not None:
        boundary = preprocessing_params.sim_area
        assert len(boundary) == 6
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device="cuda")
        for i in range(3):
            mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

        unselected_pos = init_pos[~mask, :]
        unselected_cov = init_cov[~mask, :]
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]

        rotated_pos = rotated_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_shs = init_shs[mask, :]
        init_semantic_feature = init_semantic_feature[mask, :]
        sel_gs_indices = sel_gs_indices[mask]

    # Transform the part of the 3DGS reconstruction to be simualated (`sim_area`) to the unit cube
    factor = preprocessing_params.get('scale_factor', 0.95)
    transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos, factor)
    transformed_pos = shift2center05(transformed_pos)

    # modify covariance matrix accordingly
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    #### Obtain per-Gaussian material properties by matching SAM features
    if material_preds is not None:
        per_gaussian_material_params = transfer_material_params(init_semantic_feature, material_preds, material_params)
        if preprocessing_params.visualize_material_assignment:
            create_assignment_vis_3dgs(gaussians, per_gaussian_material_params["best_match_idx"], os.path.join(export_path, "assignment_vis_3dgs.ply"), selected_mask=sel_gs_indices)
    else:
        per_gaussian_material_params = {}

    temp_gs_num = transformed_pos.shape[0]
    filling_params = preprocessing_params.particle_filling
    if filling_params is not None:
        print("Filling internal particles...")
        pos = fill_particles(
            pos=transformed_pos,
            opacity=init_opacity,
            cov=init_cov,
            grid_n=filling_params["n_grid"],
            max_samples=filling_params["max_particles_num"],
            grid_dx= 1.0 / filling_params["n_grid"],
            density_thres=filling_params["density_threshold"],
            search_thres=filling_params["search_threshold"],
            max_particles_per_cell=filling_params["max_particles_per_cell"],
            search_exclude_dir=filling_params["search_exclude_direction"],
            ray_cast_dir=filling_params["ray_cast_direction"],
            boundary=filling_params["boundary"],
            smooth=filling_params["smooth"],
        ).cuda()
        print(f'Exporting filled particles to {os.path.join(export_path, "filled_particles.ply")}')
        particle_position_tensor_to_ply(pos, os.path.join(export_path, "filled_particles.ply"))

    # Transfer visual properties of the Gaussian
    if filling_params is not None and filling_params["visualize"] == True:
        # FIXME: This is broken now since the spherical harmonics (shs) are 3 dimensional. Need to flatten and unflatten.
        properties = [init_shs, init_cov, init_opacity]
        prop_dim = [p.shape[1] for p in properties]

        props = torch.cat(properties, dim=1)
        new_props = init_filled_particles(pos[:temp_gs_num], props, pos[temp_gs_num:])
        shs, cov, opacity = new_props.split(prop_dim, dim=1)
    else:
        if filling_params is None:
            pos = transformed_pos
        cov = torch.zeros((pos.shape[0], 6), device='cuda')
        cov[:temp_gs_num] = init_cov
        shs = torch.zeros((pos.shape[0], 16, 3), device='cuda')
        shs[:temp_gs_num] = init_shs
        # Set opacity for filler particles to 0 so they are invisible
        opacity = torch.zeros((pos.shape[0], 1), device='cuda')
        opacity[:temp_gs_num] = init_opacity

    # Assign material properties of "filler" Gaussians based on closest existing Gaussian
    if filling_params is not None and material_preds is not None:
        _logger.info("Assigning material properties to filler Gaussians...")
        prop_dims = [p.shape[1] for p in per_gaussian_material_params.values()]
        props = torch.cat(list(per_gaussian_material_params.values()), dim=1)
        props = init_filled_particles(pos[:temp_gs_num], props, pos[temp_gs_num:])
        split_props = props.split(prop_dims, dim=1)
        # We make use of the fact that dictionaries have deterministic order.
        per_gaussian_material_params = {k: v for k, v in zip(per_gaussian_material_params.keys(), split_props)}

    mpm_params = {
        'pos': pos,
        'cov': cov,
        'opacity': opacity,
        'shs': shs,
        **per_gaussian_material_params,
    }
    unselected_params = {
        'pos': unselected_pos,
        'cov': unselected_cov,
        'opacity': unselected_opacity,
        'shs': unselected_shs,
    }
    if not getattr(preprocessing_params, "visualize_unselected", False):
        unselected_params["opacity"] = torch.zeros_like(unselected_params["opacity"])

    translate_params = {
        'rotation_matrices': rotation_matrices,
        'scale_origin': scale_origin,
        'original_mean_pos': original_mean_pos
    }
    return mpm_params, unselected_params, translate_params, init_screen_points



def convert_SH(
    shs_view,
    viewpoint_camera,
    pc: GaussianModel,
    position: torch.tensor,
    rotation: torch.tensor = None,
):
    shs_view = shs_view.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
    dir_pp = position - viewpoint_camera.camera_center.repeat(shs_view.shape[0], 1)
    if rotation is not None:
        n = rotation.shape[0]
        dir_pp[:n] = torch.matmul(rotation, dir_pp[:n].clone().unsqueeze(2)).squeeze(2) # replace inplace operation for backward

    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    return colors_precomp

def export_rendering(rendering, step, folder, height = None, width = None):
    os.makedirs(folder, exist_ok=True)

    cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    if height is None or width is None:
        height = cv2_img.shape[0] // 2 * 2
        width = cv2_img.shape[1] // 2 * 2
    cv2.imwrite(
        os.path.join(folder, f"{step}.png".rjust(8, "0")),
        255 * cv2_img,
    )

def save_video(folder, output_filename, start=0, end=9999, fps=30):
    
    filenames = os.listdir(folder)
    filenames = [f for f in filenames if int(f.split('.')[0]) >= start and int(f.split('.')[0]) < end]
    filenames = sorted(filenames)
    
    image = []
    for filename in filenames:
        if filename.endswith('.png'):
            image.append(imageio.v2.imread(os.path.join(folder, filename)))
    imageio.mimsave(output_filename, image, fps=fps)

def interpolate_rgb(rgb1, rgb2, t):
    return rgb1 * (1 - t) + rgb2 * t

def render_mpm_gaussian(
        model_path, pipeline, render_params,
        step, 
        viewpoint_center_worldspace, observant_coordinates,
        gaussians, background, 
        pos, cov, shs, opacity, rot,
        screen_points, 
        logits=None
    ):
    current_camera = get_camera_view(
                    model_path,
                    default_camera_index=render_params.default_camera_index,
                    center_view_world_space=viewpoint_center_worldspace - 0.3, # TODO: 0.3 is a magic number during early development, should be removed
                    observant_coordinates=observant_coordinates,
                    show_hint=render_params.show_hint,
                    init_azimuthm=render_params.init_azimuthm,
                    init_elevation=render_params.init_elevation,
                    init_radius=render_params.init_radius,
                    move_camera=render_params.move_camera,
                    current_frame=step,
                    delta_a=render_params.delta_a,
                    delta_e=render_params.delta_e,
                    delta_r=render_params.delta_r,
                )
    rasterize = initialize_resterize(
        current_camera, gaussians, pipeline, background
    )
    if logits is None:
        colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
    else:
        vis = torch.softmax(logits, dim=1)
        vis = vis[:, 1] - vis[:, 0]
        vis = (vis - vis.min()) / (vis.max() - vis.min())
        rgb1 = torch.tensor([255,0,0], device='cuda').float() / 255
        rgb2 = torch.tensor([0,0,255], device='cuda').float() / 255
        colors_precomp = interpolate_rgb(rgb1, rgb2, vis.unsqueeze(1))

    semantic_feature = torch.zeros((pos.shape[0], 256), device='cuda')
    color = rasterize(
        means3D=pos,
        means2D=screen_points,
        shs=None,
        colors_precomp=colors_precomp,
        semantic_feature=semantic_feature,
        opacities=opacity,
        scales=None,
        rotations=None,
        cov3D_precomp=cov,
    )[0]
    return color

def particle_position_tensor_to_ply(position_tensor, filename):
    # position is (n,3)
    if os.path.exists(filename):
        os.remove(filename)
    position = position_tensor.clone().detach().cpu().numpy()
    num_particles = (position).shape[0]
    position = position.astype(np.float32)
    with open(filename, "wb") as f:  # write binary
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_particles}
property float x
property float y
property float z
end_header
"""
        f.write(str.encode(header))
        f.write(position.tobytes())