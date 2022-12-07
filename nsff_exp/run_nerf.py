import os, sys
import numpy as np
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import cv2
from kornia import create_meshgrid
from tqdm import tqdm, trange

from render_utils import *
from run_nerf_helpers import *
from load_llff import *
from evaluation import *
# For performance profiling
import nvtx
import contextlib

# Experiment tracking
from clearml import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)
DEBUG = False

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                         help='input data directory')

    parser.add_argument("--render_lockcam_slowmo", action='store_true', 
                        help='render fixed view + slowmo')
    parser.add_argument("--render_slowmo_bt", action='store_true', 
                        help='render space-time interpolation')
    parser.add_argument("--final_height", type=int, default=288, 
                        help='training image height, default is 512x288')
    # training options
    parser.add_argument("--N_iters", type=int, default=360000,
                        help='number of training iterations')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=300, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*128, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*128, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_bt", action='store_true', 
                        help='render bullet time')

    parser.add_argument("--render_test", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    parser.add_argument("--target_idx", type=int, default=10, 
                        help='target_idx')
    parser.add_argument("--num_extra_sample", type=int, default=512, 
                        help='num_extra_sample')
    parser.add_argument("--decay_depth_w", action='store_true', 
                        help='decay depth weights')
    parser.add_argument("--use_motion_mask", action='store_true', 
                        help='use motion segmentation mask for hard-mining data-driven initialization')
    parser.add_argument("--decay_optical_flow_w", action='store_true', 
                        help='decay optical flow weights')

    parser.add_argument("--w_depth",   type=float, default=0.04, 
                        help='weights of depth loss')
    parser.add_argument("--w_optical_flow", type=float, default=0.02, 
                        help='weights of optical flow loss')
    parser.add_argument("--w_sm", type=float, default=0.1, 
                        help='weights of scene flow smoothness')
    parser.add_argument("--w_sf_reg", type=float, default=0.1, 
                        help='weights of scene flow regularization')
    parser.add_argument("--w_cycle", type=float, default=0.1, 
                        help='weights of cycle consistency')
    parser.add_argument("--w_prob_reg", type=float, default=0.1, 
                        help='weights of disocculusion weights')

    parser.add_argument("--w_entropy", type=float, default=1e-3, 
                        help='w_entropy regularization weight')

    parser.add_argument("--decay_iteration", type=int, default=50, 
                        help='data driven priors decay iteration * 1000')

    parser.add_argument("--chain_sf", action='store_true', 
                        help='5 frame consistency if true, \
                             otherwise 3 frame consistency')

    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=50)

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=1000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=1000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')

    # ClearML Integration Options
    parser.add_argument('--use_clearml', action='store_true',
                        help='use ClearML for experiment tracking')
    parser.add_argument("--dataset_project", type=str, default='CMPT-985',
                        help='ClearML project where dataset is to be found')
    parser.add_argument("--dataset_name", type=str, default='fern',
                        help='name of ClearML dataset')

    # CMPT-985 Term Project Options...
    parser.add_argument('--nerf_model', type=str, default='PyTorch',
                        help='NeRF architecture. Either Pytorch, FusedMLP, or CutlassMLP')
    parser.add_argument("--allow_tf32", action='store_true',
                        help='Enable TF32 tensor cores for matrix multiplication')
    parser.add_argument("--use_amp", action='store_true',
                        help='Use automated mixed-precision')
    parser.add_argument("--use_loss_autoscaler", action='store_true',
                        help='Use loss auto-scaler')
    parser.add_argument("--enable_fused_adam", action='store_true',
                        help='Enable fused kernel for Adam optimization - default False')
    parser.add_argument("--enable_pinned_memory", action='store_true',
                        help='Use pinned memory with data loaders')
    parser.add_argument("--optimizer", type=str, default='Adam',
                        help='Optimizer to use. Either SGD, Adagrad, or Adam (default)')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Up-front performance changes
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Load data
    if args.use_clearml:
        datadir = Dataset.get(dataset_name=args.dataset_name, dataset_project=args.dataset_project).get_local_copy()
    else:
        datadir = args.datadir

    if args.dataset_type == 'llff':
        target_idx = args.target_idx
        images, depths, masks, poses, bds, \
        render_poses, ref_c2w, motion_coords = load_llff_data(datadir,
                                                            args.start_frame, args.end_frame,
                                                            args.factor,
                                                            target_idx=target_idx,
                                                            recenter=True, bd_factor=.9,
                                                            spherify=args.spherify, 
                                                            final_height=args.final_height)


        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, datadir)
        i_test = []
        i_val = [] #i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.percentile(bds[:, 0], 5) * 0.8 #np.ndarray.min(bds) #* .9
            far = np.percentile(bds[:, 1], 95) * 1.1 #np.ndarray.max(bds) #* 1.
        else:
            near = 0.
            far = 1.

        print('NEAR FAR', near, far)
    else:
        print('ONLY SUPPORT LLFF!!!!!!!!')
        sys.exit()


    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create log dir and copy the config file
    basedir = args.basedir
    args.expname = args.expname + '_F%02d-%02d'%(args.start_frame, args.end_frame)
    
    # args.expname = args.expname + '_sigma_rgb-%.2f'%(args.sigma_rgb) \
                # + '_use-rgb-w_' + str(args.use_rgb_w) + '_F%02d-%02d'%(args.start_frame, args.end_frame)
    
    expname = args.expname

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }

    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)


    if args.render_bt:
        print('RENDER VIEW INTERPOLATION')
        
        render_poses = torch.Tensor(render_poses).to(device)
        print('target_idx ', target_idx)

        num_img = float(poses.shape[0])
        img_idx_embed = target_idx/float(num_img) * 2. - 1.0

        testsavedir = os.path.join(basedir, expname, 
                                'render-spiral-frame-%03d'%\
                                target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            render_bullet_time(render_poses, img_idx_embed, num_img, hwf, 
                               args.chunk, render_kwargs_test, 
                               gt_imgs=images, savedir=testsavedir, 
                               render_factor=args.render_factor)

        return

    if args.render_lockcam_slowmo:
        print('RENDER TIME INTERPOLATION')

        num_img = float(poses.shape[0])
        ref_c2w = torch.Tensor(ref_c2w).to(device)
        print('target_idx ', target_idx)

        testsavedir = os.path.join(basedir, expname, 'render-lockcam-slowmo')
        os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            render_lockcam_slowmo(args, ref_c2w, num_img, hwf,
                            render_kwargs_test,
                            gt_imgs=images, savedir=testsavedir,
                            target_idx=target_idx)

            return 

    if args.render_slowmo_bt:
        print('RENDER SLOW MOTION')

        curr_ts = 0
        render_poses = poses #torch.Tensor(poses).to(device)
        bt_poses = create_bt_poses(hwf) 
        bt_poses = bt_poses * 10

        with torch.no_grad():

            testsavedir = os.path.join(basedir, expname, 
                                    'render-slowmo_bt_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            images = torch.Tensor(images)#.to(device)

            print('render poses shape', render_poses.shape)
            render_slowmo_bt(depths, render_poses, bt_poses, 
                            hwf, args.chunk, render_kwargs_test,
                            gt_imgs=images, savedir=testsavedir, 
                            render_factor=args.render_factor, 
                            target_idx=10)
            # print('Done rendering', i,testsavedir)

        return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand

    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    depths = torch.Tensor(depths).to(device)
    masks = 1.0 - torch.Tensor(masks).to(device)
    poses = torch.Tensor(poses).to(device)

    N_iters = args.N_iters
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    uv_grid = create_meshgrid(H, W, normalized_coordinates=False)[0].cuda() # (H, W, 2)

    # Summary writers
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    num_img = float(images.shape[0])
    
    decay_iteration = max(args.decay_iteration, 
                          args.end_frame - args.start_frame)
    decay_iteration = min(decay_iteration, 250)

    chain_bwd = 0

    # Throttle update intervals for ClearML reporting
    mininterval = 60 if args.use_clearml else 0.1
    maxinterval = 60 if args.use_clearml else 10.0

    # Note: bfloat16 has the same range as float32, at the cost of precision.
    if args.use_amp:
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_context = contextlib.nullcontext()

    if args.use_loss_autoscaler:
        loss_scaler = torch.cuda.amp.GradScaler()

    start = start + 1
    for i in trange(start, N_iters, mininterval=mininterval, maxinterval=maxinterval):
        with nvtx.annotate(f'Training iteration {i}'):

            # Note: bfloat16 has the same range as float32, at the cost of precision.
            with autocast_context:
                chain_bwd = 1 - chain_bwd
                time0 = time.time()

                # Random from one image
                img_i = np.random.choice(i_train)

                if i % (decay_iteration * 1000) == 0:
                    torch.cuda.empty_cache()

                target = images[img_i]
                pose = poses[img_i, :3,:4]
                depth_gt = depths[img_i]
                hard_coords = torch.Tensor(motion_coords[img_i])
                mask_gt = masks[img_i]

                # TODO: this seems like something which could be cached instead of read every iteration
                with nvtx.annotate("Read Optical Flow"):
                    if img_i == 0:
                        flow_fwd, fwd_mask = read_optical_flow(datadir, img_i,
                                                            args.start_frame, fwd=True)
                        flow_bwd, bwd_mask = np.zeros_like(flow_fwd), np.zeros_like(fwd_mask)
                    elif img_i == num_img - 1:
                        flow_bwd, bwd_mask = read_optical_flow(datadir, img_i,
                                                            args.start_frame, fwd=False)
                        flow_fwd, fwd_mask = np.zeros_like(flow_bwd), np.zeros_like(bwd_mask)
                    else:
                        flow_fwd, fwd_mask = read_optical_flow(datadir,
                                                            img_i, args.start_frame, 
                                                            fwd=True)
                        flow_bwd, bwd_mask = read_optical_flow(datadir,
                                                            img_i, args.start_frame, 
                                                            fwd=False)

                    flow_fwd = torch.Tensor(flow_fwd)
                    fwd_mask = torch.Tensor(fwd_mask)
                
                    flow_bwd = torch.Tensor(flow_bwd)
                    bwd_mask = torch.Tensor(bwd_mask)

                    # more correct way for flow loss
                    flow_fwd = flow_fwd + uv_grid
                    flow_bwd = flow_bwd + uv_grid

                with nvtx.annotate("Get Rays"):
                    rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
                    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)

                    if args.use_motion_mask and i < decay_iteration * 1000:
                        num_extra_sample = args.num_extra_sample
                        select_inds_hard = np.random.choice(hard_coords.shape[0], 
                                                            size=[min(hard_coords.shape[0], 
                                                                num_extra_sample)], 
                                                            replace=False)  # (N_rand,)
                        select_inds_all = np.random.choice(coords.shape[0], 
                                                        size=[N_rand], 
                                                        replace=False)  # (N_rand,)

                        select_coords_hard = hard_coords[select_inds_hard].long()
                        select_coords_all = coords[select_inds_all].long()

                        select_coords = torch.cat([select_coords_all, select_coords_hard], 0)

                    else:
                        select_inds = np.random.choice(coords.shape[0], 
                                                    size=[N_rand], 
                                                    replace=False)  # (N_rand,)
                        select_coords = coords[select_inds].long()  # (N_rand, 2)
                    
                    rays_o = rays_o[select_coords[:, 0], 
                                    select_coords[:, 1]]  # (N_rand, 3)
                    rays_d = rays_d[select_coords[:, 0], 
                                    select_coords[:, 1]]  # (N_rand, 3)
                    batch_rays = torch.stack([rays_o, rays_d], 0)
                    target_rgb = target[select_coords[:, 0], 
                                        select_coords[:, 1]]  # (N_rand, 3)
                    target_depth = depth_gt[select_coords[:, 0], 
                                        select_coords[:, 1]]
                    target_mask = mask_gt[select_coords[:, 0], 
                                        select_coords[:, 1]].unsqueeze(-1)

                    target_of_fwd = flow_fwd[select_coords[:, 0], 
                                            select_coords[:, 1]]
                    target_fwd_mask = fwd_mask[select_coords[:, 0], 
                                            select_coords[:, 1]].unsqueeze(-1)#.repeat(1, 2)

                    target_of_bwd = flow_bwd[select_coords[:, 0], 
                                            select_coords[:, 1]]
                    target_bwd_mask = bwd_mask[select_coords[:, 0], 
                                            select_coords[:, 1]].unsqueeze(-1)#.repeat(1, 2)

                img_idx_embed = img_i/num_img * 2. - 1.0

                #####  Core optimization loop  #####
                if args.chain_sf and i > decay_iteration * 1000 * 2:
                    chain_5frames = True
                else:
                    chain_5frames = False

                ret = render(img_idx_embed, 
                            chain_bwd, 
                            chain_5frames,
                            num_img, H, W, focal, 
                            chunk=args.chunk, 
                            rays=batch_rays,
                            verbose=i < 10, retraw=True,
                            **render_kwargs_train)

                pose_post = poses[min(img_i + 1, int(num_img) - 1), :3,:4]
                pose_prev = poses[max(img_i - 1, 0), :3,:4]

                render_of_fwd, render_of_bwd = compute_optical_flow(pose_post, 
                                                                    pose, pose_prev, 
                                                                    H, W, focal, 
                                                                    ret)

                weight_map_post = ret['prob_map_post']
                weight_map_prev = ret['prob_map_prev']

                weight_post = 1. - ret['raw_prob_ref2post']
                weight_prev = 1. - ret['raw_prob_ref2prev']
                prob_reg_loss = args.w_prob_reg * (torch.mean(torch.abs(ret['raw_prob_ref2prev'])) \
                                        + torch.mean(torch.abs(ret['raw_prob_ref2post'])))

                # dynamic rendering loss
                if i <= decay_iteration * 1000:
                    # dynamic rendering loss
                    render_loss = img2mse(ret['rgb_map_ref_dy'], target_rgb)
                    render_loss += compute_mse(ret['rgb_map_post_dy'], 
                                            target_rgb, 
                                            weight_map_post.unsqueeze(-1))
                    render_loss += compute_mse(ret['rgb_map_prev_dy'], 
                                            target_rgb, 
                                            weight_map_prev.unsqueeze(-1))
                else:
                    weights_map_dd = ret['weights_map_dd'].unsqueeze(-1).detach()

                    # dynamic rendering loss
                    render_loss = compute_mse(ret['rgb_map_ref_dy'], 
                                            target_rgb, 
                                            weights_map_dd)
                    render_loss += compute_mse(ret['rgb_map_post_dy'], 
                                            target_rgb, 
                                            weight_map_post.unsqueeze(-1) * weights_map_dd)
                    render_loss += compute_mse(ret['rgb_map_prev_dy'], 
                                            target_rgb, 
                                            weight_map_prev.unsqueeze(-1) * weights_map_dd)

                # union rendering loss
                render_loss += img2mse(ret['rgb_map_ref'][:N_rand, ...], 
                                    target_rgb[:N_rand, ...])

                sf_cycle_loss = args.w_cycle * compute_mae(ret['raw_sf_ref2post'], 
                                                        -ret['raw_sf_post2ref'], 
                                                        weight_post.unsqueeze(-1), dim=3) 
                sf_cycle_loss += args.w_cycle * compute_mae(ret['raw_sf_ref2prev'], 
                                                            -ret['raw_sf_prev2ref'], 
                                                            weight_prev.unsqueeze(-1), dim=3)
                
                # regularization loss
                render_sf_ref2prev = torch.sum(ret['weights_ref_dy'].unsqueeze(-1) * ret['raw_sf_ref2prev'], -1)
                render_sf_ref2post = torch.sum(ret['weights_ref_dy'].unsqueeze(-1) * ret['raw_sf_ref2post'], -1)

                sf_reg_loss = args.w_sf_reg * (torch.mean(torch.abs(render_sf_ref2prev)) \
                                            + torch.mean(torch.abs(render_sf_ref2post))) 

                divsor = i // (decay_iteration * 1000)

                decay_rate = 10

                if args.decay_depth_w:
                    w_depth = args.w_depth/(decay_rate ** divsor)
                else:
                    w_depth = args.w_depth

                if args.decay_optical_flow_w:
                    w_of = args.w_optical_flow/(decay_rate ** divsor)
                else:
                    w_of = args.w_optical_flow

                depth_loss = w_depth * compute_depth_loss(ret['depth_map_ref_dy'], -target_depth)

                if img_i == 0:
                    flow_loss = w_of * compute_mae(render_of_fwd, 
                                                target_of_fwd, 
                                                target_fwd_mask)
                elif img_i == num_img - 1:
                    flow_loss = w_of * compute_mae(render_of_bwd, 
                                                target_of_bwd, 
                                                target_bwd_mask)
                else:
                    flow_loss = w_of * compute_mae(render_of_fwd, 
                                                target_of_fwd, 
                                                target_fwd_mask)
                    flow_loss += w_of * compute_mae(render_of_bwd, 
                                                target_of_bwd, 
                                                target_bwd_mask)

                # scene flow spatial smoothness loss
                # Equation 1 in supplementary material
                scene_flow_smoothness_loss = args.w_sm * \
                    compute_scene_flow_spatial_smoothness(ret['raw_pts_ref'], ret['raw_pts_post'], H, W, focal)

                scene_flow_smoothness_loss += args.w_sm * \
                    compute_scene_flow_spatial_smoothness(ret['raw_pts_ref'], ret['raw_pts_prev'], H, W, focal)

                # scene flow temporal smoothness loss
                # Equation 2 in supplementary material
                scene_flow_smoothness_loss += args.w_sm * \
                    compute_scene_flow_temporal_smoothness(ret['raw_pts_ref'], ret['raw_pts_post'], ret['raw_pts_prev'], H, W, focal)

                scene_flow_smoothness_loss += args.w_sm * \
                    compute_scene_flow_temporal_smoothness(ret['raw_pts_ref'], ret['raw_pts_post'], ret['raw_pts_prev'], H, W, focal)

                entropy_loss = args.w_entropy * torch.mean(-ret['raw_blend_w'] * torch.log(ret['raw_blend_w'] + 1e-7))

                # # ======================================  two-frames chain loss ===============================
                if chain_bwd:
                    scene_flow_smoothness_loss += args.w_sm * \
                        compute_scene_flow_temporal_smoothness(ret['raw_pts_prev'], ret['raw_pts_ref'], ret['raw_pts_pp'], H, W, focal)

                else:
                    scene_flow_smoothness_loss += args.w_sm * \
                        compute_scene_flow_temporal_smoothness(ret['raw_pts_post'], ret['raw_pts_pp'], ret['raw_pts_ref'], H, W, focal)

                if chain_5frames:
                    render_loss += compute_mse(ret['rgb_map_pp_dy'], 
                                            target_rgb, 
                                            weights_map_dd)


                loss = sf_reg_loss + sf_cycle_loss + \
                    render_loss + flow_loss + \
                    scene_flow_smoothness_loss + prob_reg_loss + \
                    depth_loss + entropy_loss

            optimizer.zero_grad()
            if args.use_loss_autoscaler:
                loss_scaler.scale(loss).backward()
                loss_scaler.step(optimizer)
                loss_scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            ################################

            # Rest is logging

            # Write scalars to log
            if i % args.i_print == 0 and i > 0:
                torch.cuda.synchronize()

                writer.add_scalar("loss", loss.item(), i)
                
                writer.add_scalar("render_loss", render_loss.item(), i)
                writer.add_scalar("depth_loss", depth_loss.item(), i)
                writer.add_scalar("flow_loss", flow_loss.item(), i)
                writer.add_scalar("disocclusion_loss", prob_reg_loss.item(), i)
                writer.add_scalar("sf_reg_loss", sf_reg_loss.item(), i)
                writer.add_scalar("sf_cycle_loss", sf_cycle_loss.item(), i)
                writer.add_scalar("scene_flow_smoothness_loss", scene_flow_smoothness_loss.item(), i)

            # Save network weights
            if i%args.i_weights==0 and i > 0:
                path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))

                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_rigid': render_kwargs_train['network_rigid'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)

                print('Saved checkpoints at', path)

            # Generate and save images (RGB, depth map, etc) to log
            if i%args.i_img == 0 and i > 0:
                # img_i = np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                target_depth = depths[img_i] - torch.min(depths[img_i])

                # img_idx_embed = img_i/num_img * 2. - 1.0

                # if img_i == 0:
                #     flow_fwd, fwd_mask = read_optical_flow(datadir, img_i,
                #                                            args.start_frame, fwd=True)
                #     flow_bwd, bwd_mask = np.zeros_like(flow_fwd), np.zeros_like(fwd_mask)
                # elif img_i == num_img - 1:
                #     flow_bwd, bwd_mask = read_optical_flow(datadir, img_i,
                #                                            args.start_frame, fwd=False)
                #     flow_fwd, fwd_mask = np.zeros_like(flow_bwd), np.zeros_like(bwd_mask)
                # else:
                #     flow_fwd, fwd_mask = read_optical_flow(datadir,
                #                                            img_i, args.start_frame, 
                #                                            fwd=True)
                #     flow_bwd, bwd_mask = read_optical_flow(datadir,
                #                                            img_i, args.start_frame, 
                #                                            fwd=False)

                # flow_fwd_rgb = torch.Tensor(flow_to_image(flow_fwd)/255.)#.cuda()
                # writer.add_image("val/gt_flow_fwd", 
                #                 flow_fwd_rgb, global_step=i, dataformats='HWC')
                # flow_bwd_rgb = torch.Tensor(flow_to_image(flow_bwd)/255.)#.cuda()
                # writer.add_image("val/gt_flow_bwd", 
                #                 flow_bwd_rgb, global_step=i, dataformats='HWC')
                with torch.no_grad():
                    ret = render(img_idx_embed, 
                                chain_bwd, False,
                                num_img, H, W, focal, 
                                chunk=1024*16, 
                                c2w=pose,
                                **render_kwargs_test)
                    
                    model = models.PerceptualLoss(model='net-lin',net='alex',
                                    use_gpu=True,version=0.1)

                    #evaluation metrics 
                    gt_img = target.cpu().numpy()
                    rgb = ret['rgb_map_ref'].detach().cpu().numpy()
                    mask_gt_eval = mask_gt.unsqueeze(1).repeat(1, 3, 1).reshape([mask_gt.shape[0], mask_gt.shape[1], 3])
                    mask_gt_eval = mask_gt_eval.cpu().numpy()
                    
                    ssim = calculate_ssim(gt_img, rgb, mask_gt_eval)
                    psnr = calculate_psnr(gt_img, rgb, mask_gt_eval)
                

                    target_0 = torch.Tensor(gt_img[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))
                    rgb_0 = torch.Tensor(rgb[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))
                    mask_gt_0 = torch.Tensor(mask_gt_eval[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

                    lpips = model.forward(target_0, rgb_0, mask_gt_0).item()
                    
                    # pose_post = poses[min(img_i + 1, int(num_img) - 1), :3,:4]
                    # pose_prev = poses[max(img_i - 1, 0), :3,:4]
                    # render_of_fwd, render_of_bwd = compute_optical_flow(pose_post, pose, pose_prev, 
                    #                                                     H, W, focal, ret, n_dim=2)

                    # render_flow_fwd_rgb = torch.Tensor(flow_to_image(render_of_fwd.cpu().numpy())/255.)#.cuda()
                    # render_flow_bwd_rgb = torch.Tensor(flow_to_image(render_of_bwd.cpu().numpy())/255.)#.cuda()
                    writer.add_scalar("ssim", ssim, i)
                    writer.add_scalar("psnr", psnr, i)
                    writer.add_scalar("lpips", lpips, i)

                    writer.add_image("predicted_rgb", torch.clamp(ret['rgb_map_ref'], 0., 1.),
                                    global_step=i, dataformats='HWC')
                    writer.add_image("predicted_depth", normalize_depth(ret['depth_map_ref']),
                                    global_step=i, dataformats='HW')

                    writer.add_image("static_model_rgb", torch.clamp(ret['rgb_map_rig'], 0., 1.),
                                    global_step=i, dataformats='HWC')
                    writer.add_image("static_model_depth", normalize_depth(ret['depth_map_rig']),
                                    global_step=i, dataformats='HW')

                    writer.add_image("dynamic_model_rgb", torch.clamp(ret['rgb_map_ref_dy'], 0., 1.),
                                    global_step=i, dataformats='HWC')
                    writer.add_image("dynamic_model_depth", normalize_depth(ret['depth_map_ref_dy']),
                                    global_step=i, dataformats='HW')

                    # writer.add_image("val/rgb_map_pp_dy", torch.clamp(ret['rgb_map_pp_dy'], 0., 1.), 
                                    # global_step=i, dataformats='HWC')

                    writer.add_image("ground_truth_rgb", target,
                                    global_step=i, dataformats='HWC')
                    writer.add_image("ground_truth_monocular_disp",
                                    torch.clamp(target_depth /percentile(target_depth, 97), 0., 1.),
                                    global_step=i, dataformats='HW')

                    writer.add_image("static_dynamic_weights",
                                    ret['weights_map_dd'],
                                    global_step=i,
                                    dataformats='HW')

            global_step += 1

if __name__=='__main__':
    # Set the default tensor type - keep it at 32-bit float as much as possible.
    # Only use 16-bit for the model weights and training data
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
