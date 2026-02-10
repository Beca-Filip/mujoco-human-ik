# Motion Capture to MuJoCo Inverse Kinematics Pipeline


## Overview

This system implements a full pipeline for retargeting human motion capture (MoCap) data onto a MuJoCo human model using inverse kinematics (IK). The pipeline loads raw MoCap marker data, preprocesses it (offset correction, unit conversion, scaling, filtering), and then solves inverse kinematics frame-by-frame to produce a joint position (qpos) trajectory that can be simulated or rendered.

The architecture is modular: each processing step is isolated into its own module (data loading, preprocessing, IK, filtering, visualization), making the system easier to debug, extend, and reuse.

---

## High-Level Pipeline

1. Load motion capture data
2. Preprocess MoCap data (offsets, unit conversion, scaling)
3. Visualize raw MoCap skeleton and trajectories
4. Load and initialize MuJoCo model
5. Compute axis scaling between MoCap and MuJoCo model
6. Configure marker-to-site mapping
7. Run inverse kinematics for each frame
8. Store and visualize joint trajectories

---

## System Architecture (Diagram)

+-------------------+
|  MoCap TSV File   |
+---------+---------+
          |
          v
+-------------------------+
| Load MoCap data         |
+-------------------------+
          |
          v
+-------------------------+
| Preprocessing           |
+-------------------------+
          |
          v
+-------------------------+
| Load MuJoCo model       |
+-------------------------+
          |
          v
+-------------------------+
| Axis Scaling            |
| (compute & apply)       |
+-------------------------+
          |
          v
+-------------------------+
| Filtered Marker Targets |
+------------+------------+        
             |
             v
+-----------------------------------+
| Inverse Kinematics Solver         |
| (solve_ik_for_frame)              |
+-----------------+-----------------+
                  |
                  v
+-----------------------------------+
| qpos Trajectory           	    |
+-----------------+-----------------+
                  |
        +---------+----------+
        |                    |
        v                    v
+---------------+   +-----------------------+
| Simulation    |   | Rendering / Plotting  |
+---------------+   +-----------------------+


---

## Main Script (main())

The main() function orchestrates the entire pipeline, from loading MoCap data to simulating the resulting motion in MuJoCo.

---

## Data Loading and Preprocessing

### load_mocap_data(data_path)

Loads motion capture data from a TSV file into a pandas DataFrame.

Input:

 Path to MoCap file

Output:

 DataFrame containing 3D marker positions per frame

---

### get_names(mocap_data)

Extracts joint/marker names from the MoCap dataset.

---

### apply_offsets(mocap_data)

Applies predefined spatial offsets to markers to correct systematic sensor or calibration biases.

---

### mm_to_meters(mocap_data)

Converts all marker coordinates from millimeters to meters to match MuJoCo’s unit system.

---

## MuJoCo Model Initialization

This loads the humanoid model and initializes its state. The forward pass ensures that all derived quantities (site positions, body transforms) are valid.

Printed diagnostics:
 nq: number of generalized coordinates
 nv: number of degrees of freedom
 nsite: number of sites used for IK targets

---

## Visualization of Raw MoCap Data

### compute_axis_limits(mocap_data)

Computes shared axis limits to ensure consistent 3D visualization scaling.

### plot_skeleton_at_frame(...)

Plots a 3D skeleton at a specific frame to visually inspect marker correctness.

### plot_joint_trajectories(...)

Plots time trajectories of all joints/markers for sanity checking.

---

## Axis Scaling Between MoCap and Model

### compute_axis_scaling_factors(...)

Computes scaling factors (Y and Z axes) so that key anatomical distances (shoulders, hips, knees) match between the MoCap data and the MuJoCo model.

### apply_axis_scaling(mocap_data, y_scale, z_scale)

Applies the computed scaling factors to MoCap data.

---

## Marker-to-Site Configuration

### Site Names

A subset of anatomical landmarks is selected as IK targets:
 Shoulders, elbows, wrists, knees, ankles

### Site Weights

Each site is assigned a weight indicating its importance in the IK optimization:
 Ankles have the highest weight (10.0)
 Knees medium (2.0)
 Upper body default (1.0)

Higher weights enforce tighter tracking.

---

## Inverse Kinematics Loop

### Target Extraction

For each frame:

 Marker positions are extracted from the MoCap DataFrame

### Temporal Filtering

This reduces jitter and enforces temporal smoothness.

---

### IK Solver

This function:
 Solves a multi-site inverse kinematics problem
 Respects joint limits
 Updates data.qpos in-place

---

### Trajectory Storage

For each frame, the resulting joint configuration is saved:

Final result:
 qpos_trajectory.shape == (num_frames, model.nq)

---

## Simulation and Rendering

### simulation_qpos_trajectory(model, qpos_trajectory)

Replays the computed joint trajectory inside MuJoCo for visual validation.

### render_qpos_trajectory_to_video(...)

Renders the motion to a video file for offline analysis or presentation.

---

## Typical Use Cases
 Human motion retargeting
 Biomechanics analysis
 Robotics imitation learning
 Simulation-based animation pipelines