# /workspace/bebop_bot/bebop_training/export_bebop_model.py

import argparse
import os
import torch
import torch.nn as nn
import gymnasium as gym
from datetime import datetime

# --- STEP 1: Launch App ---
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Export Bebop Policy to C++ and ONNX.")
parser.add_argument("--task", type=str, default="Isaac-Bebop-Flat-v0", help="Task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt file")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- STEP 2: Imports ---
import isaaclab
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_onnx
import bebop_training

def generate_cpp_header(actor_model, obs_dim, action_dim, env_cfg, output_path):
    """Generates a complete C++ header with weights and inference logic."""
    print(f"[INFO] Generating C++ header: {output_path}")
    
    # Extract Action Scales
    try: 
        leg_scale = env_cfg.actions.legs_pos.scale
    except AttributeError: 
        leg_scale = 0.5

    try: 
        wheel_scale = env_cfg.actions.wheels_vel.scale
    except AttributeError: 
        wheel_scale = 20.0
    
    # Standard Observation size for Bebop (No History)
    BASE_OBS_DIM = 30 
    has_history = (obs_dim > BASE_OBS_DIM)
    
    # Clipping limits (Must match training distribution)
    clip_lin_vel = 3.0
    clip_ang_vel = 10.0
    clip_dof_vel = 15.0
    
    with open(output_path, 'w') as f:
        f.write("#ifndef BEBOP_POLICY_H\n")
        f.write("#define BEBOP_POLICY_H\n\n")
        f.write("#include <math.h>\n")
        f.write("#include <string.h>\n")
        f.write("#include <Arduino.h> // Required for PROGMEM\n\n")
        
        f.write(f"// ============================================================================\n")
        f.write(f"// Generated on: {datetime.now()}\n")
        f.write(f"// Architecture: {obs_dim} inputs -> {action_dim} outputs\n")
        if has_history:
            f.write(f"// ⚠ WARNING: HISTORY DETECTED! Input dim ({obs_dim}) > Base dim ({BASE_OBS_DIM})\n")
        f.write(f"// ============================================================================\n\n")

        f.write("namespace bebop_policy {\n\n")

        # =================================================================
        # CONSTANTS
        # =================================================================
        f.write(f"    constexpr int OBS_DIM = {obs_dim};\n")
        f.write(f"    constexpr int ACTION_DIM = {action_dim};\n")
        f.write(f"\n")
        
        f.write(f"    // Observation Scales (1.0 = Raw Values)\n")
        f.write(f"    // We define these so main_sim.cpp compiles, even if they are 1.0\n")
        f.write(f"    constexpr float SCALE_LIN_VEL = 1.0f;\n")
        f.write(f"    constexpr float SCALE_ANG_VEL = 1.0f;\n")
        f.write(f"    constexpr float SCALE_DOF_POS = 1.0f;\n")
        f.write(f"    constexpr float SCALE_DOF_VEL = 1.0f;\n")
        f.write(f"\n")

        f.write(f"    // Action Scales\n")
        f.write(f"    constexpr float SCALE_ACTION_LEGS = {leg_scale:.4f}f;\n")
        f.write(f"    constexpr float SCALE_ACTION_WHEELS = {wheel_scale:.4f}f;\n")
        f.write(f"\n")
        
        f.write(f"    // Observation Clipping (Safety)\n")
        f.write(f"    constexpr float CLIP_LIN_VEL = {clip_lin_vel:.4f}f;\n")
        f.write(f"    constexpr float CLIP_ANG_VEL = {clip_ang_vel:.4f}f;\n")
        f.write(f"    constexpr float CLIP_DOF_VEL = {clip_dof_vel:.4f}f;\n\n")

        # =================================================================
        # WEIGHTS
        # =================================================================
        layer_sizes = []
        param_names = [] 

        valid_params = []
        for name, param in actor_model.named_parameters():
            if "critic" in name or "std" in name: continue
            valid_params.append((name, param))

        for name, param in valid_params:
            flat_data = param.data.cpu().numpy().flatten()
            clean_name = name.replace('actor.', '').replace('.', '_')
            var_name = "layer_" + clean_name
            param_names.append(var_name)

            f.write(f"    const float {var_name}[] PROGMEM = {{\n        ")
            for i, val in enumerate(flat_data):
                f.write(f"{val:.8f}f, ")
                if (i+1) % 10 == 0: f.write("\n        ")
            f.write("\n    };\n\n")
            
            if "bias" in name:
                layer_sizes.append(param.shape[0])

        # =================================================================
        # INFERENCE CLASS
        # =================================================================
        f.write("    class Policy {\n")
        f.write("    private:\n")
        for i, size in enumerate(layer_sizes[:-1]): 
            f.write(f"        float buffer_l{i}[{size}];\n")
        
        f.write("\n        inline float elu(float x) { return (x > 0) ? x : (expf(x) - 1.0f); }\n\n")
        
        f.write("        void dense(const float* in, const float* w, const float* b, float* out, int in_size, int out_size, bool activate) {\n")
        f.write("            for(int i=0; i<out_size; i++) {\n")
        f.write("                float val = b[i];\n")
        f.write("                for(int j=0; j<in_size; j++) val += in[j] * w[i*in_size + j];\n")
        f.write("                out[i] = activate ? elu(val) : val;\n")
        f.write("            }\n")
        f.write("        }\n\n")

        f.write("    public:\n")
        f.write("        void init() {}\n\n")
        
        f.write("        void infer(const float* obs, float* actions) {\n")
        
        current_input = "obs"
        current_in_dim = f"{obs_dim}"
        buffer_idx = 0
        
        for i in range(0, len(param_names), 2):
            weight_name = param_names[i]
            bias_name = param_names[i+1]
            is_last_layer = (i == len(param_names) - 2)
            
            if is_last_layer:
                output_ptr = "actions"
                out_dim = layer_sizes[buffer_idx]
                activation = "false"
            else:
                output_ptr = f"buffer_l{buffer_idx}"
                out_dim = layer_sizes[buffer_idx]
                activation = "true"
            
            f.write(f"            dense({current_input}, {weight_name}, {bias_name}, {output_ptr}, {current_in_dim}, {out_dim}, {activation});\n")
            
            current_input = output_ptr
            current_in_dim = f"{out_dim}"
            buffer_idx += 1
        
        f.write("        }\n")
        f.write("    };\n")
        f.write("\n} // namespace bebop_policy\n")
        f.write("#endif\n")
        
    print(f"[SUCCESS] C++ Header generated: {output_path}")

def find_actor_critic(runner):
    alg = runner.alg
    if hasattr(alg, 'actor_critic'): return alg.actor_critic
    elif hasattr(alg, 'policy'): return alg.policy
    for attr in dir(alg):
        a = getattr(alg, attr)
        if isinstance(a, nn.Module) and hasattr(a, 'actor'): return a
    raise RuntimeError("Could not find ActorCritic module.")

def find_actor_module(actor_critic):
    if hasattr(actor_critic, 'actor'): return actor_critic.actor
    return actor_critic

def main():
    checkpoint_path = args.checkpoint
    run_dir = os.path.dirname(checkpoint_path)
    
    print("=" * 70)
    print("BEBOP POLICY EXPORT TOOL")
    print("=" * 70)
    
    # 1. Setup
    task_spec = gym.spec(args.task)
    env_cfg = task_spec.kwargs.get("env_cfg_entry_point")()
    agent_cfg = task_spec.kwargs.get("rsl_rl_cfg_entry_point")()
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg, render_mode=None))
    
    # 2. Load
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=run_dir, device="cpu")
    runner.load(checkpoint_path)
    
    # 3. Extract
    actor_critic = find_actor_critic(runner)
    actor_critic.eval()
    actor_model = find_actor_module(actor_critic)
    
    linear_layers = [m for m in actor_model.modules() if isinstance(m, nn.Linear)]
    obs_dim = linear_layers[0].in_features
    action_dim = linear_layers[-1].out_features
    
    # 4. Export ONNX (This generates policy.onnx)
    print(f"[INFO] Exporting ONNX to {run_dir}...")
    onnx_path = os.path.join(run_dir)
    export_policy_as_onnx(actor_critic, onnx_path, filename="policy.onnx", verbose=False)
    print(f"[SUCCESS] ONNX Exported: {os.path.join(onnx_path, 'policy.onnx')}")
    
    # 5. Export C++
    header_path = os.path.join(run_dir, "BebopPolicy.h")
    generate_cpp_header(actor_model, obs_dim, action_dim, env_cfg, header_path)
    
    print("=" * 70)
    print("EXPORT COMPLETE")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()