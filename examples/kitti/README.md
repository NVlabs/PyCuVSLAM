# Tutorial: Running PyCuVSLAM on the KITTI dataset

The KITTI Vision Benchmark Suite is a high-quality dataset benchmarking and comparing various computer vision algorithms.
Among other options, the KITTI dataset has sequences for evaluating stereo-visual odometry

## Dataset Setup

1. To get the KITTI test sequences, download 
[the odometry data set](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) (grayscale, 22 GB).
You will have to register for an account. Once you do, you will receive a download link to the `data_odometry_gray.zip` file.

2. Unpack it to your dataset folder:

    ```bash
    unzip data_odometry_gray.zip -d examples/kitti
    ```

    When you unpack it, you get the
    following structure:
    ```
    dataset_odometry_gray
       dataset  -> should be mapped or placed in examples/kitti/dataset
          sequences
              00
                 image_0
                    000000.png
                      ...
                    004550.png  
                 image_1
                    000000.png
                      ...
                    004550.png  
                 calib.txt
                 times.txt
              01  
                 ...
    ```
    Each of the 20 folders has a stereo sequence and calibration of cameras.

3. Ensure the dataset path is correctly set at the beginning of the visual tracking scripts.

## Running PyCuVSLAM Stereo Visual Odometry

Execute the following command to start tracking on KITTI:

```bash
python3 track_kitti.py
```

The visualization should appear in rerun, demonstrating estimated trajectory, visual features selected on left stereo image and their 3D positions:

![Visualization Example](../../assets/tutorial_kitti.jpg)

## SLAM Mapping: Collecting, Storing, Loading, and Localization

PyCuVSLAM provides SLAM functionality, enabling you to map an environment, save the generated map to disk, reload it, and then localize within the saved map. It complements odometry by serving as an additional visual SLAM tracker. The SLAM tracker refines the pose estimation whenever visual features in the current frame match features previously stored in map keyframes — a process known as _loop closure detection_. Loop closure helps to significantly reduce drift that accumulates in odometry over time. However, when a loop closure occurs, you might notice sudden corrections ("jumps") in the SLAM trajectory due to adjustments made by the SLAM algorithm.

Below is an example demonstrating loop closure detection. When the vehicle revisits a previously mapped area, loop closure events are triggered (indicated by large red dots), and the SLAM pose is adjusted based on matching visual features stored in the SLAM map. From this point onward, a difference appears between the instantaneous odometry pose (purple line) and the SLAM-corrected pose (yellow line):

![Loop Closure Demonstration](../../assets/tutorial_kitti_lc.gif)

To enable SLAM in PyCuVSLAM, you must provide a SLAM configuration when initializing the tracker:

```python
odom_cfg = cuvslam.Tracker.OdometryConfig(...)
slam_cfg = cuvslam.Tracker.SlamConfig(sync_mode=...)
tracker = cuvslam.Tracker(cuvslam.Rig(...), odom_cfg, slam_cfg)
```

SLAM can operate in two modes:

- **Asynchronous (`sync_mode=False`, recommended for real-time applications):**  
  SLAM processing runs in a separate, non-blocking thread, allowing visual odometry to continue uninterrupted.

- **Synchronous (`sync_mode=True`):**  
  SLAM processing runs in the same thread as odometry, causing visual tracking to pause at each keyframe until all SLAM-related processing completes. This can cause noticeable delays (seconds-long pauses), making it unsuitable for real-time camera-based applications.

Now, each call to `tracker.track` will return both an odometry and SLAM poses:

```python
odometry_pose_estimate, slam_pose = tracker.track(...)
```

You can try the provided example demonstrating SLAM mode using the KITTI dataset:

```bash
python3 track_kitti_slam.py
```

At the end of the mapping process, the trajectory and map will be saved as `trajectory_tum.txt` and `map/data.mdb`, respectively.

> **Note:** The entire `map` directory will be overwritten, so make sure it doesn't contain important data.

### Localization with an Existing SLAM Map

After the mapping stage, you can reuse the previously stored map and trajectory to perform localization from an arbitrary starting point. For example, you may start from frame index `IDX = 700`. PyCuVSLAM will use the corresponding stereo pair and pose from the saved trajectory as an initial guess for localization.

To use SLAM localization, first configure localization parameters using `cuvslam.Tracker.SlamLocalizationSettings`. Then, pass these settings to the method `tracker.localize_in_map`. After initiating localization, PyCuVSLAM will search for matching visual features in the loaded map near the provided initial pose guess. If localization succeeds, you will see a confirmation message on the terminal, and the new initial coordinates will be updated accordingly.

As a result, the initial pose displayed will reflect the localized position (no longer zero translation or unit rotation). Additionally, loop closure detection events will trigger immediately, as the complete map from the mapping stage is already loaded:

![Map loading and localization Demonstration](../../assets/tutorial_slam_map_load.gif)

## Running PyCuVSLAM with input masks

PyCuVSLAM supports both static and dynamic masks for input images, which exclude features from specified image regions during visual tracking. 

### Static Masks

Static masks can be easily defined as border offsets during `cuvslam.Camera` initialization:

```python
camera = cuvslam.Camera()
camera.border_top = 50
camera.border_bottom = 150
camera.border_left = 30
camera.border_right = 30
```

Static masks are especially helpful in the following situations:

1. **Irrelevant objects in the camera’s field of view**: parts of the image include objects that should not influence camera motion estimation (such as the vehicle static hood, sensor mounts, or  spinning drone rotors). See the example below.
2. **Images with geometric distortion**: unrectified images may introduce significant distortions near the frame edges, which can impact visual tracking accuracy. Refer to the example in the [OAK-D camera stereo odometry tutorial](../oak-d/README.md#static-masks-to-improve-visual-tracking).
3. **Invalid depth regions**: depth estimation at the frame edges is unreliable, often due to field of view misalignment with the RGB camera or after rectification. See the example using the [TUM RGB-D dataset](../tum/README.md#masking-regions-to-prevent-feature-selection).

The example below from [Oxford Robotcar dataset](https://robotcar-dataset.robots.ox.ac.uk/) demonstrates a common use case where a car's hood appears in the camera's field of view. In the first stream, no static masks are applied, resulting in 2D features being incorrectly selected on the hood. In the second stream, static masks are applied to the bottom portion of the image frame, ensuring features are only selected from the actual scene.

![No mask case](../../assets/robotcar_no_mask.gif)

![Static mask case](../../assets/robotcar_static_mask.gif)

### Dynamic Masks with PyTorch Tensors

For complex scenes or dynamic objects, you can provide binary masks to specify regions of interest (ROIs) to exclude from visual tracking. These masks can be dynamically generated in real-time using deep neural networks.

> **PyTorch GPU Tensor Support:** To optimize performance and avoid unnecessary type conversions or GPU I/O overhead, PyCuVSLAM accepts both PyTorch GPU tensors and NumPy arrays as input for images and masks.

![Mask Demonstration](../../assets/tutorial_kitti_mask.gif)

### Example: Real-Time Car Segmentation

This example demonstrates how PyCuVSLAM stereo visual odometry can be combined with real-time car segmentation on the KITTI dataset. We use the [NVIDIA Segformer model from HuggingFace](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/tree/main) to dynamically identify and exclude car ROIs from PyCuVSLAM feature detection.

#### Prerequisites

Install the following deep learning packages:

```bash
torch==2.7.0+cu126
torchaudio==2.7.0+cu126
torchvision==0.22.0+cu126
transformers==4.52.3
```

Run the example script to visualize the estimated trajectory, left stereo images, selected features, and dynamic car masks detected by the Segformer model:

```bash
python3 track_kitti_masks.py
```

**Performance Tip:** To reduce computational overhead, run PyCuVSLAM in `Tracker.MulticameraMode.Performance` mode and provide masks only for the left-camera images. In this mode, PyCuVSLAM selects visual features exclusively from the left-camera images for pose estimation from provided stereo-pair