# TrackForge
# Multi-Object Tracking with Kalman Filtering

A Python-based multi-object tracking system combining YOLO11 object detection, Kalman filtering, Euclidean-distance association, and the Hungarian algorithm on the KITTI autonomous-driving dataset.

## Overview

Object detection identifies objects within an individual image, but it does not inherently maintain their identities across successive frames. Multi-object tracking addresses this limitation by associating detections over time and estimating how objects move through a scene.

This project implements a multi-object tracking pipeline that combines deep-learning-based object detection with classical state estimation and assignment algorithms. YOLO11 generates object detections, while individual Kalman filters estimate the motion of tracked objects. A distance-based cost matrix and the Hungarian algorithm associate new detections with existing tracks.

The implementation is designed to demonstrate the core mechanics of multi-object tracking without relying on a dedicated tracking framework.

### Key Components

* **Object detection:** YOLO11 detects supported object categories in KITTI camera frames.
* **State estimation:** A Kalman filter predicts and updates each track's position and velocity.
* **Data association:** Euclidean distance measures the separation between predicted track positions and detected object centres.
* **Assignment optimisation:** The Hungarian algorithm determines the minimum-cost assignment between tracks and detections.
* **Association gating:** A configurable distance threshold rejects implausible matches.
* **Track lifecycle management:** Unmatched detections can initiate new tracks, while temporarily unmatched tracks are retained for a limited number of frames.
* **LiDAR-camera projection:** KITTI calibration parameters are used to project LiDAR points into the camera image for depth-related processing.

## System Architecture

The processing pipeline follows this sequence:

1. Load a camera frame and its corresponding LiDAR scan.
2. Run YOLO11 inference on the camera frame.
3. Filter detections by object category and confidence.
4. Project LiDAR points into camera-image coordinates.
5. Predict the states of existing tracks.
6. Calculate distances between predicted track centres and detected object centres.
7. Apply distance gating and solve the assignment problem using the Hungarian algorithm.
8. Update matched tracks and initialise tracks for unmatched detections.
9. Retain temporarily unmatched tracks until their allowed prediction period expires.
10. Visualise the tracking results and write the processed frames to a video.

The detector supplies object measurements, the Kalman filter provides temporal state estimation, and the assignment stage maintains track continuity across frames.

## Mathematical Foundation

### 1. State Representation

Each tracked object is represented by a four-dimensional state vector:

$$
\mathbf{x}_t =
\begin{bmatrix}
x_t \\
y_t \\
v_{x,t} \\
v_{y,t}
\end{bmatrix}
$$

where:

* $x_t, y_t$ represent the estimated object-centre coordinates in image pixels.
* $v_{x,t}, v_{y,t}$ represent the corresponding horizontal and vertical velocities.

The centre coordinates are obtained from object detections. Velocity is estimated through successive prediction and correction steps.

### 2. Motion Model

The tracker uses a constant-velocity motion model:

$$
\mathbf{x}_{t|t-1} = F\,\mathbf{x}_{t-1|t-1}
$$

where the transition matrix is:

$$
F =
\begin{bmatrix}
1 & 0 & \Delta t & 0 \\
0 & 1 & 0 & \Delta t \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

Here, $\Delta t$ denotes the interval between successive measurements in the chosen time units.

The model assumes that velocity remains approximately constant over a single prediction interval. Actual object motion may deviate from this assumption, so subsequent measurements are used to correct the prediction.

### 3. Prediction Step

The Kalman filter propagates the state estimate and its uncertainty:

$$
\hat{\mathbf{x}}_{t|t-1} = F\,\hat{\mathbf{x}}_{t-1|t-1}
$$

$$
P_{t|t-1} = F\,P_{t-1|t-1}\,F^{T} + Q
$$

where:

* $P$ is the state-estimation covariance matrix.
* $Q$ represents process noise.
* The predicted state estimates where the object is expected to appear in the next frame.

The covariance evolves throughout tracking, representing changes in uncertainty as the filter predicts and incorporates measurements.

### 4. Measurement Model

The detector provides the object's centre coordinates rather than its velocity. The measurement model is therefore:

$$
\mathbf{z}_t = H\,\mathbf{x}_t + \mathbf{v}_t
$$

with:

$$
H =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0
\end{bmatrix}
$$

Here, $\mathbf{z}_t$ is the measured object centre and $\mathbf{v}_t$ represents measurement noise.

### 5. Correction Step

When a detection is associated with a track, the filter corrects its predicted state.

The innovation is:

$$
\mathbf{y}_t = \mathbf{z}_t - H\,\hat{\mathbf{x}}_{t|t-1}
$$

The Kalman gain is:

$$
K_t = P_{t|t-1}\,H^{T}\left(H\,P_{t|t-1}\,H^{T} + R\right)^{-1}
$$

The corrected state becomes:

$$
\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + K_t\,\mathbf{y}_t
$$

The covariance is updated using the implemented covariance correction:

$$
P_{t|t} = (I - K_t H)\,P_{t|t-1}
$$

Here, $R$ represents measurement noise. The Kalman gain determines how the predicted state and new measurement contribute to the corrected estimate.

### 6. Data Association

Each frame may contain multiple detections and multiple existing tracks. The system must determine which detections correspond to which tracks.

For predicted track $i$ and detection $j$, the assignment cost is the Euclidean distance between their centres:

$$
C_{ij} = \sqrt{(\hat{x}_i - x_j)^2 + (\hat{y}_i - y_j)^2}
$$

The resulting cost matrix is passed to the Hungarian algorithm, implemented through SciPy's `linear_sum_assignment`.

The algorithm finds a minimum-cost one-to-one assignment between the available rows and columns.

#### Distance Gating

Before assignment, costs exceeding the configured distance threshold are replaced with a large penalty:

$$
C_{ij}^{\text{gated}} =
\begin{cases}
C_{ij}, & C_{ij} \leq d_{\max} \\
M, & C_{ij} > d_{\max}
\end{cases}
$$

where $d_{\max}$ is the maximum permitted association distance and $M$ is a large penalty.

The original distances are checked again before updating tracks, so an assignment beyond the permitted threshold is rejected.

This approach aligns association with the current Kalman state, which models object-centre position and velocity rather than bounding-box dimensions.

## Track Lifecycle

The tracking manager handles the creation, updating, temporary retention, and removal of tracks.

**Track initialisation:** A detection that has no valid association can initialise a new tracker. The track starts from the measured centre position.

**Track prediction:** Existing trackers predict their states before the current detections are associated.

**Track correction:** Matched trackers incorporate the associated detection to refine their state estimates.

**Temporary unmatched tracks:** When a track has no corresponding detection, its prediction can be retained for a limited number of frames. This helps the system tolerate short detection gaps.

**Track removal:** Tracks exceeding the configured coasting limit are removed from the active tracker collection.

These mechanisms provide continuity when detections are intermittent, although prolonged occlusion or similar-looking objects can still cause tracking errors or identity switches.

## LiDAR-Camera Projection

The project also processes KITTI LiDAR measurements using the dataset's calibration parameters.

LiDAR points are transformed into the camera coordinate system and projected into image coordinates. Points behind the camera or outside the image bounds are filtered out.

The projected points are used to associate spatial depth information with detected image regions. The implementation uses the available depth values for distance-related processing.

This component complements the image-based tracking pipeline; the current Kalman state itself remains defined in two-dimensional image coordinates.

## Dataset

The project uses the KITTI Raw dataset, specifically the sequence:

`2011_09_26_drive_0011_sync`

The required data includes:

* Synchronized and rectified camera images.
* Corresponding LiDAR scans.
* Camera calibration parameters.
* IMU-to-LiDAR and LiDAR-to-camera calibration files.

The expected directory structure is:

```text
data/
└── kitti/
    └── 2011_09_26/
        ├── calib_cam_to_cam.txt
        ├── calib_imu_to_velo.txt
        ├── calib_velo_to_cam.txt
        └── 2011_09_26_drive_0011_sync/
            ├── image_00/
            ├── image_01/
            ├── image_02/
            ├── image_03/
            ├── oxts/
            └── velodyne_points/
```

The directory names must match the paths configured in the execution script.

## Installation

### Prerequisites

* Python 3.11
* pip
* Git
* A compatible PyTorch and TorchVision installation
* An NVIDIA GPU and compatible CUDA-enabled PyTorch build for GPU inference; CPU inference may also be used

### 1. Clone the repository

```bash
git clone https://github.com/gsactown30/multi-frame-object-tracking-with-kalman-filtering.git
cd multi-frame-object-tracking-with-kalman-filtering
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

Install the dependencies listed in `requirements.txt`.

PyTorch and TorchVision should be installed as a compatible pair using the appropriate official PyTorch installation instructions for the target system.

### 4. Prepare the dataset

Download the required KITTI Raw sequence and calibration files. Place them in the directory structure described above.

### 5. Run the tracker

After the dataset and dependencies are configured, run:

```bash
python run_tracking.py
```

The processed video is written to the configured output location.

## Configuration

The tracking behaviour is controlled by parameters in the tracking execution script and manager.

| Parameter         | Purpose                                                                  |
| ----------------- | ------------------------------------------------------------------------ |
| `confThreshold`   | Minimum detector confidence accepted by the pipeline                     |
| `distThreshold`   | Maximum centre distance allowed for a track-detection association        |
| `coastThreshold`  | Maximum number of prediction-only frames allowed before removing a track |
| `measureVariance` | Measurement-noise parameter supplied to the Kalman tracker               |
| `classVariance`   | Object-class-specific process-noise configuration                        |

The current implementation uses a confidence threshold of `0.5`, a distance threshold of `100` pixels, and a coasting threshold of `15` frames. These are implementation settings, not universal optimal values.

## Limitations

* The Kalman motion model assumes approximately constant velocity over each prediction interval.
* Association uses image-space centre distance and does not directly model bounding-box scale.
* The fixed association threshold may not be equally suitable for all object speeds, frame intervals, or scene conditions.
* Long occlusions and crowded scenes can result in missed associations or identity switches.
* The tracking state is two-dimensional even though LiDAR measurements are used for depth-related processing.
* Tracking quality depends on detector performance, calibration accuracy, and the selected parameters.

## Future Improvements

Potential directions for further development include:

* Evaluating association methods that incorporate predicted bounding-box overlap.
* Introducing appearance embeddings to help distinguish objects with similar motion.
* Investigating motion uncertainty and adaptive association thresholds.
* Extending the tracking state into three-dimensional space.
* Evaluating identity consistency, detection association quality, and tracking performance using established MOT metrics.

These are possible extensions rather than features claimed by the current implementation.

## References

1. Kalman, R. E. (1960). *A New Approach to Linear Filtering and Prediction Problems.* Journal of Basic Engineering, 82(1), 35–45.
2. Kuhn, H. W. (1955). *The Hungarian Method for the Assignment Problem.* Naval Research Logistics Quarterly, 2(1–2), 83–97.
3. Geiger, A., Lenz, P., Stiller, C., and Urtasun, R. (2013). *Vision meets Robotics: The KITTI Dataset.* The International Journal of Robotics Research.
4. Bewley, A. et al. (2016). *Simple Online and Realtime Tracking.* IEEE International Conference on Image Processing.
5. [Ultralytics Documentation](https://docs.ultralytics.com/)
6. [SciPy Linear Assignment Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)
7. [KITTI Raw Data](https://www.cvlibs.net/datasets/kitti/raw_data.php)

---

**Project focus:** Object detection, Kalman-based state estimation, gated data association, and multi-object tracking for autonomous-driving perception.
