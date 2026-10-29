# MoCap to MuJoCo IK Pipeline

![Simulation](https://raw.githubusercontent.com/Beca-Filip/mujoco-human-ik/main/sim_gif.gif)

[![PyPI - Version](https://img.shields.io/pypi/v/mujoco-human-ik-package.svg)](https://pypi.org/project/mujoco-human-ik-package/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mujoco-human-ik-package.svg)](https://pypi.org/project/mujoco-human-ik-package/)

-----

Retargets human motion capture data onto a MuJoCo humanoid model via inverse kinematics. The package supports single-trial processing, batch processing, running simulation, joint trajectory visualization and tracking error evaluation.

## Table of Contents

- [Installation](#installation)
- [License](#license)
- [Dependencies](#dependencies)
- [Features](#features)
- [Outputs](#outputs)
- [Citation](#citation)

## Installation

```console
pip install mujoco-human-ik-package
```

## License

`mujoco-human-ik-package` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Dependencies

Major dependencies:
- MuJoCo
- NumPy
- Pandas
- SciPy
- Matplotlib

## Features

### Single-Trial Inverse Kinematics
MoCap → MuJoCo inverse kinematics pipeline.


```console
from mujoco_human_ik_package import one_trial_ik
```

The pipeline:
1. Load MoCap TSV data
2. Preprocessing and data validation, apply offsets, convert mm → m, scale axes
3. Generates or loads a subject-specific MuJoCo model
4. Map anatomical markers to MuJoCo sites (shoulders, elbows, wrists, knees, ankles)
5. Solve IK per frame with site weights (ankles: 10.0, knees: 2.0, upper body: 1.0)
6. Exports the resulting qpos trajectory
7. Optionally renders a video, runs a simulation, and performs inverse dynamics

Notes:
- input MoCap format:
The function expects a motion capture TSV file containing the following marker names:
l_hip_pos, r_hip_pos, l_knee_pos, r_knee_pos, l_ankle_pos, r_ankle_pos, l_metatarsal_pos, r_metatarsal_pos, l_shoulder_pos, r_shoulder_pos, l_elbow_pos, r_elbow_pos, l_wrist_pos, r_wrist_pos.
Each marker must contain 3D coordinates (x, y, z).
All values are assumed to be in millimeters (mm) and are internally converted to meters (m) during preprocessing.
- if an existing MuJoCo XML model is not provided (model_path=None), the model is automatically generated.
In that case, the following subject parameters must be provided:
subj_height (meters), 
subj_mass (kilograms), 
subj_sex ("male"/"female").
These parameters are validated and required to construct a valid subject-specific biomechanical model.
- the MuJoCo XML model must contain the following site names for marker mapping:
greater_trochanter_left, greater_trochanter_right, lateral_femoral_epicondyle_left, lateral_femoral_epicondyle_right, lateral_maleollus_left, lateral_maleollus_right, metatarsal_fifth_left, metatarsal_fifth_right, acromion_left, acromion_right, lateral_humeral_epicondyle_left, lateral_humeral_epicondyle_right, ulnar_styloid_left, ulnar_styloid_right.
These sites are used for anatomical correspondence between motion capture markers and the MuJoCo model.
- optional execution flags:
	- print_flag - Enables verbose logging during execution
	- save_plot_flag - Saves generated plots to disk
	- sim_flag - Runs MuJoCo simulation visualization
	- sim_pause_flag - Starts simulation in paused mode
	- inverse_dynamics_flag - Enables inverse dynamics computation (Note: currently experimental and expected to be improved in future versions)
	- output_video_path - Enables rendering and saving of simulation video
	
Example:
```console
one_trial_ik(mocap_path="trial.tsv", subj_height=1.6, subj_mass=60, subj_sex="female")
```

### Batch Processing
Process an entire directory of MoCap trials.

```console
from mujoco_human_ik_package import run_batch
```

Loads every TSV file in directory, loads subject-specific anthropometric information, validates subject data and runs the inverse kinematics pipeline for each trial.

For every successfully processed trial, the generated outputs may include:
- joint position trajectories (qpos)
- inverse dynamics results
- simulation videos
- subject-specific MuJoCo XML models

Notes:
- the file with subject info is expected to contain the following columns:
  ID, name, height [m], mass [kg] and gender ['male'/'female']
- subject height and mass values are validated before processing
- trials with missing subject information or invalid anthropometric
values are skipped and recorded in the error text file
- same assuming for MoCap as for Single-Trial Inverse Kinematics

Example:
```console
run_batch(mocap_path_to_dir='Trials', subj_info='subjects.csv')
```

### Running Simulation
This module provides a MuJoCo-based simulation using a predefined XML model
and a sequence of qpos states stored in a CSV file.

```console
from mujoco_human_ik_package import run_simulation
```

The pipeline:
1. Loads a MuJoCo model from an XML file
2. Loads qpos (joint configuration) trajectory from a CSV file
3. Runs a forward simulation step to initialize the model state
4. Visualizes the trajectory using simulation_qpos_trajectory

Notes:
The simulation can be run with pauses.

Example:
```console
run_simulation(mj_model_path='human.xml', qpos_path='trial_qpos.csv', pause_flag=True)
```

### Joint Trajectory Visualization 
Processes and visualizes joint angle trajectories from multiple qpos CSV files.

```console
from mujoco_human_ik_package import qpos_plot
```

The pipeline:
1. Loads all qpos CSV files from a directory
2. Extracts hip, knee and ankle joint angles
3. Detects movement phase based on center of mass (COM) trajectory
4. Normalize movement duration
5. Aggregates statistics across trials (mean, min, max, std)
6. Plots joint angle envelopes with anatomical limits

Notes:
The function assumes that:
- sampling frequency of 300 Hz (FS constant)
- consistent column naming across all CSV files (format '<left/right>_joint_axis')
- Movement segmentation is based on COM peak detection
- Output angles are converted to degrees before plotting

Example:
```console
qpos_plot(qpos_path_dir='Results/qpos')
```

### Tracking Error Evaluation
Compute global marker tracking errors for a set of qpos trajectories. For each CSV file in the provided directory, the function loads the corresponding MuJoCo model and motion capture data, reconstructs marker positions from the model, and computes the Euclidean distance error between simulated and measured marker locations.

```console
from mujoco_human_ik_package import mean_err
```

The following metrics are calculated:
- Root Mean Square Error (RMSE)
- Median Euclidean distance error
- 95th percentile (P95) distance error

Notes:
Errors are expressed in centimeters.
The function assumes that:
- Subject-specific MuJoCo models are available as
  `human_<subject_id>.xml`
- Motion capture files are located in the `data` directory
- CSV filenames follow the pattern
  `<qpos>_<subject>_<jump>_<trial>.csv`

Example:
```console
mean_err(qpos_path_dir='Results/qpos')
```

## Outputs
This package may create following directories and files:
- human.xml - generated MuJoCo XML model for the corresponding subject
- failed_mocaps.txt - TXT file containing error messages or explanation for any MoCap files that could not be processed
- Results/ 
	- qpos/ - directory for storing qpos trajectories in CSV format
	- Clean_MoCap/ - directory containing plots of interpolated MoCap data
	- Videos/ - directory for storing rendered simulation videos
	- GRF/ - optional directory for storing Ground Reaction Force plots generated during inverse dynamics



## Citation

```bibtex
@software{becanovic2026mocap,
  author  = {Be{\v{c}}anovi{\'{c}}, Filip and Svilar, Marina},
  title   = {{MoCap to MuJoCo Inverse Kinematics Pipeline}},
  year    = {2026},
  url     = {https://github.com/Beca-Filip/mujoco-human-ik}
}
```
