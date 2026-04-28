# MoCap to MuJoCo IK Pipeline

Retargets human motion capture data onto a MuJoCo humanoid model via inverse kinematics. The pipeline preprocesses raw TSV marker data (offset correction, unit conversion, axis scaling, filtering) and solves IK frame-by-frame to produce a `qpos` trajectory for simulation or rendering.

## Pipeline

1. Load MoCap TSV data
2. Apply offsets, convert mm → m, scale axes to match MuJoCo model
3. Map anatomical markers to MuJoCo sites (shoulders, elbows, wrists, knees, ankles)
4. Solve IK per frame with site weights (ankles: 10.0, knees: 2.0, upper body: 1.0)
5. Store `qpos` trajectory → simulate or render to video

## Usage

```python
python main.py
```

## Citation

```bibtex
@software{becanovic2026mocap,
  author  = {Be{\v{c}}anovi{\'{c}}, Filip and Svilar, Marina},
  title   = {{MoCap to MuJoCo Inverse Kinematics Pipeline}},
  year    = {2026},
  url     = {https://github.com/Beca-Filip/mujoco-human-ik}
}
```
