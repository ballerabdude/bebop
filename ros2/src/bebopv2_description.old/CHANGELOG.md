# Changelog

All notable changes to the `bebopv2_description` package are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] – 2026-05-23

### Summary

Reworked the leg kinematic chain to fix incorrect joint nomenclature and ship
the redesigned hip flexion assembly from the latest CAD revision.

The previous tree mislabeled the joint immediately below `base_link` as
"hip abduction" when it was in fact the flexion (pitch) axis. The chain has
been renamed end-to-end so the joint names now match their actual anatomical
function. The first leg segment off `base_link` is also a new physical part,
not just a rename – it carries new mass, inertia, and mesh geometry.

### New kinematic chain

```
base_link
  └─ hip_flexion_{left,right}_joint   (revolute, pitch, Y axis)   [was: hip_abduction_*_joint]
       └─ hip_flexion_{left,right}_1                              [redesigned link; new mesh + inertia]
            └─ hip_abduction_{left,right}_joint  (revolute, roll, X axis)  [was: femur_*_joint]
                 └─ hip_abduction_{left,right}_1                 [was: femur_*_1; mesh + inertia unchanged]
                      └─ knee_flexion_{left,right}_joint  (revolute, pitch, Y axis)  [was: shin_*_joint]
                           └─ knee_flexion_{left,right}_1        [was: shin_*_1; mesh + inertia unchanged]
                                └─ foot_{left,right}_joint  (revolute, pitch, Y axis)
                                     └─ foot_{left,right}_1
```

### Changed – Links

| Old name                       | New name                       | Notes                                                                 |
|--------------------------------|--------------------------------|-----------------------------------------------------------------------|
| `hip_abduction_{left,right}_1` | `hip_flexion_{left,right}_1`   | **Renamed and redesigned.** New mesh, mass 1.000 kg → 1.1494 kg, new inertia tensor and CoM. |
| `femur_{left,right}_1`         | `hip_abduction_{left,right}_1` | Renamed only. Mesh, mass (1.7420 kg) and inertia carried over.        |
| `shin_{left,right}_1`          | `knee_flexion_{left,right}_1`  | Renamed only. Mesh, mass (0.1137 kg) and inertia carried over.        |
| `foot_{left,right}_1`          | `foot_{left,right}_1`          | Unchanged.                                                            |
| `base_link`                    | `base_link`                    | Mass refined from CAD: 7.0 kg → 6.70 kg; inertia tensor rescaled accordingly. Mesh unchanged. |

### Changed – Joints

| Old name                          | New name                          | Notes                                                                                     |
|-----------------------------------|-----------------------------------|-------------------------------------------------------------------------------------------|
| `hip_abduction_{left,right}_joint`| `hip_flexion_{left,right}_joint`  | Renamed to reflect actual pitch-axis function. Parent/child updated to new links. Limits unchanged: ±0.785398 rad (±45°). |
| `femur_{left,right}_joint`        | `hip_abduction_{left,right}_joint`| Renamed. Origin moved from `(0.02955, ±0.07, 0)` to `(0.02955, 0, -0.14)` to match the new hip_flexion link geometry. Limits tightened (see below). |
| `shin_{left,right}_joint`         | `knee_flexion_{left,right}_joint` | Renamed; parent link updated to new `hip_abduction_*_1`. Origin and limits unchanged.    |
| `foot_{left,right}_joint`         | `foot_{left,right}_joint`         | Unchanged. Parent link reference updated to `knee_flexion_*_1`.                          |

#### Joint limit changes (`hip_abduction_*_joint`, formerly `femur_*_joint`)

| Side  | Old upper / lower (rad) | New upper / lower (rad) |
|-------|-------------------------|-------------------------|
| Right | +0.349066 / −0.785398   | +0.174533 / −0.349066   |
| Left  | +0.785398 / −0.349066   | +0.349066 / −0.174533   |

Travel range halved on both sides to reflect the mechanical hard stops of the
redesigned hip flexion bracket.

#### Hip flexion mount offsets (from `base_link`)

- Right: `(-0.003527, -0.1495, 0.013)` → `(-0.003527, -0.1503, 0.013)`
- Left:  `(-0.003789,  0.1505, 0.013132)` (unchanged)

### Added – Meshes

- `meshes/hip_flexion_left_1.stl` (≈2.76 MB) – new redesigned part.
- `meshes/hip_flexion_right_1.stl` (≈2.76 MB) – new redesigned part.
- `meshes/knee_flexion_left_1.stl` (≈282 KB) – renamed from `shin_left_1.stl`.
- `meshes/knee_flexion_right_1.stl` (≈282 KB) – renamed from `shin_right_1.stl`.

### Changed – Meshes

- `meshes/hip_abduction_left_1.stl` and `meshes/hip_abduction_right_1.stl`
  now contain the **former `femur_*_1.stl` geometry** (≈7.75 MB each). The
  filename is reused; the geometry it points at is the long thigh segment, not
  the old triangular hip bracket.

### Removed – Meshes

- `meshes/femur_left_1.stl`, `meshes/femur_right_1.stl` (content moved to
  `hip_abduction_*_1.stl`).
- `meshes/shin_left_1.stl`, `meshes/shin_right_1.stl` (content moved to
  `knee_flexion_*_1.stl`).
- The previous `hip_abduction_*_1.stl` (≈3.08 MB triangular hip bracket) is
  superseded by the redesigned `hip_flexion_*_1.stl` parts.

### Updated files

- `urdf/bebopv2.xacro` – all link names, joint names, parent/child references,
  origins, and inertials updated to match the new chain.
- `urdf/bebopv2.gazebo` – `<gazebo reference="…">` entries updated for the new
  link names (`hip_flexion_*_1`, `hip_abduction_*_1`, `knee_flexion_*_1`).

### Migration notes for downstream consumers

Anything that referred to the old joint or link names needs updating. The
direct rename map is:

```
hip_abduction_{left,right}_joint  →  hip_flexion_{left,right}_joint
femur_{left,right}_joint          →  hip_abduction_{left,right}_joint
shin_{left,right}_joint           →  knee_flexion_{left,right}_joint
foot_{left,right}_joint           →  foot_{left,right}_joint               (unchanged)

hip_abduction_{left,right}_1      →  hip_flexion_{left,right}_1            (also redesigned)
femur_{left,right}_1              →  hip_abduction_{left,right}_1
shin_{left,right}_1               →  knee_flexion_{left,right}_1
foot_{left,right}_1               →  foot_{left,right}_1                    (unchanged)
```

Known dependents that likely need to be updated in lock-step:

- `ros2/src/bebopv2_description/config/` – any controller / joint-list YAMLs.
- `ros2/src/bebopv2_description/urdf/bebopv2.ros2control` – joint references.
- `ros2/src/bebopv2_description/launch/` – any joint name remappings.
- `firmware/bebop-linux/config/bebop_v2.yaml` – joint ordering / names.
- `sim/usd/bebopv2/` – USD physics payload (also referenced in the old base_link inertia comment).
- `bebop-app` Motor Bench / telemetry views that hard-code joint labels.

### Rationale

The previous naming was inherited from an earlier CAD revision in which the
top hip link was modeled as a simple abduction bracket. In the current
mechanical design the joint immediately under the torso is the pitch (flexion)
axis, and the abduction (roll) axis lives one link down inside the hip
assembly. Aligning the URDF names with the actual anatomy avoids a
long-running source of confusion in controls code, telemetry, and any
human-facing labels.
