# Depth camera presets and filters (measurements)

Measured 2026-10-22 on the **near** Gemini 335Lg (848×480 @ 30, SDK 2.9.3),
robot still, 2.5 s per set, with the depth tuning tool that was removed with
the navigation stack on 2026-10-22
(`git log -- bebop-vision/tools/orbbec_depth_tune.py`). Scores: valid pixel
fraction, per-pixel temporal noise on floor pixels by range band, and
**flicker**: floor pixels that read as an obstacle (≥ 0.10 m) or a drop within
1.2 m in a given frame.

| preset / filters | valid | floor noise 0.5-1 / 1-1.7 m | phantom obstacles per frame (max) |
|---|---|---|---|
| High Accuracy, none | 87.4 % | 1.5 / 2.5 mm | 0 (0) |
| High Accuracy, temporal | 87.5 % | 0.8 / 1.3 mm | 0 (0) |
| **High Accuracy, noise + temporal** | 87.5 % | 0.7 / 1.3 mm | **0 (0)** |
| High Accuracy, noise + temporal + hole | 97.8 % | 0.8 / 1.3 mm | 3.9 (13) |
| High Accuracy, temporal + hole (deployed) | 97.8 % | 0.8 / 1.3 mm | 7.1 (24) |
| High Density, noise + temporal | 92.9 % | 0.6 / 1.1 mm | 0.9 (3) |
| High Density, temporal + hole | 98.2 % | 0.6 / 1.1 mm | 3.5 (7) |
| Default, temporal + hole | 97.9 % | 0.7 / 1.2 mm | 6.3 (24) |

The raw depth is quiet; **hole filling** is what invents raised points.
The near camera runs `[temporal, hole]` at the operator's request, and so
does the far camera (`depth_filters` in `bebop-vision/config/orbbec_rig.yaml`,
both on the High Accuracy preset); the per-camera rig YAML options
`depth_filters`, `depth_range_mm` and `depth_noise_params` remain for a later
decision.

**Device-side range and precision.** The Gemini 335Lg rejects
`OB_PROP_MIN_DEPTH_INT` / `OB_PROP_MAX_DEPTH_INT` and the depth precision
level (not writable, not readable), so a range limit is host-side:
`depth_range_mm: [lo, hi]` in the rig YAML becomes an SDK `ThresholdFilter`
when the device write is refused. The `depth_precision_mm` and
`depth_work_mode` rig options were removed on 2026-10-24 (the precision level
is not writable here, and a preset carries its own work mode); history:
`git log -- bebop-vision/bebop_vision/orbbec.py`. The on-device noise
removal (`max_speckle_size`, `max_diff`) is writable (`depth_noise_params:`),
but 200/400 and 128 measured no better than the preset's 80/256. Laser power
(level 4), auto-exposure and gain are writable and untested. A doll on the
floor reads as an obstacle at 0.32 m with the noise + temporal set (body
4-10 cm, head above the 10 cm obstacle floor).
