# term-project
CMPT 985 Term Project

### Installation

- Setup for Python virtual environment. I had a lot of problems with Python 3.10, so I downgraded to 3.9.15 and everything is running smoothly again. From the project root directory, execute:

  ```
  python3 -m venv venv
  source venv/bin/activate
  pip3 install -r requirements.txt
  ```


### Pre-Processing

NSFF requires a few pre-processing steps to convert a set of images into a dynamic NeRF scene. These steps are run as scripts in the `nsff_scripts` directory:

1. Create camera intrinsics and extrinsics from image data, generating the `scene.json` and `poses_bounds.npy` files:

   ```
   python ./save_poses_nerf.py --data_path "/home/xxx/datasets/kid-running/dense/"
   ```

2. Compute depth maps from monocular images. Uses the "[MiDaS](https://pytorch.org/hub/intelisl_midas_v2/)" model. Before running this script, you'll have to download the pre-trained [model.pt](https://drive.google.com/drive/folders/1G-NFZKEA8KSWojUKecpJPVoq5XCjBLOV) weights and put them in the `nsff_scripts` directory. This will create the files `dense/disp/00xxx.npy` from the input files `dense/images/00xxx.png`:

   ```
   python ./run_midas.py --data_path /home/xxx/datasets/kid-running/dense/ --resize_height 288
   ```

3. Create the optical flow files. Uses the [RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf) model. The model files can be downloaded with the `download_models.sh` script. This step will create the files `dense/flow_i1/00xxx_fwd.npz` and `dense/flow_i1/00xxx_bwd.npz` for the forward and backward optical flow:

   ```
   python ./run_flows_video.py --data_path /home/xxx/datasets/kid-running/dense/ --model models/raft-things.pth
   ```



### Training

Training and rendering steps are run as scripts in the `nsff_exp` directory. To train a scene, modify the configuration file in the `configs` directory appropriately, and execute the training run with:

```
python ./run_nerf.py --config configs/config_kid-running.txt
```

This will generate files in the `logs/${expname}` directory, with the experiment name specified in the config file. Telemetry is written in TensorBoard format in the `logs/summaries` directory.



