# experiments/exp_rough_terrain.py
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from ..envs.bebop_base_cfg import BebopBaseEnvCfg

@configclass
class BebopRoughTerrainCfg(BebopBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # OVERRIDE: Change terrain to rough generator
        self.scene.terrain = sim_utils.TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=sim_utils.TerrainGeneratorCfg(
                seed=42,
                curriculum=True, # Get harder as robot gets smarter
                difficulty_range=(0.0, 1.0),
                num_rows=10,
                num_cols=20,
                sub_terrains={
                    "rough_pyramid": sim_utils.TerrainGeneratorCfg.SubTerrainCfg(
                        proportion=1.0,
                        mix_noise_origin=0.5,
                        mix_noise_diff=0.5,
                        mix_noise_scale=0.1 # 10cm bumps
                    )
                }
            )
        )