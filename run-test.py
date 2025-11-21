import argparse
import os
import numpy as np
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

# 解析命令行参数
parser = argparse.ArgumentParser(description='Run OpenPI policy inference test')
parser.add_argument('--pytorch', action='store_true', 
                    help='Use PyTorch converted model instead of JAX model')
args = parser.parse_args()

print("1. 正在加载配置...")
config = _config.get_config("pi05_droid")

print("2. 正在检查/下载模型权重...")
if args.pytorch:
    # use the pytorch converted version
    checkpoint_dir = os.path.expanduser("~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch")
    print("   使用 PyTorch 模型")
else:
    # use the jax version
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
    print("   使用 JAX 模型")

print("3. 创建策略模型...")
policy = policy_config.create_trained_policy(config, checkpoint_dir)

print("4. 构建虚拟观测数据...")
# ---------------------------------------------------------
# 修复点：补充 Proprioception (本体感觉) 数据
# ---------------------------------------------------------

# 1. 图像数据 (不变)
dummy_exterior = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
dummy_wrist = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

# 2. 关节数据 (DROID 使用 Franka 机械臂，有7个自由度)
# 我们随机生成7个角度 (float32)
dummy_joint = np.random.randn(7).astype(np.float32)

# 3. 夹爪数据 (通常是一个浮点数，表示张开宽度，0~1之间)
# 我们假设现在是半张开状态 (0.5)
dummy_gripper = np.array([0.5], dtype=np.float32)

example = {
    "observation/exterior_image_1_left": dummy_exterior,
    "observation/wrist_image_left": dummy_wrist,
    
    # 新增的关键字段！
    "observation/joint_position": dummy_joint,
    "observation/gripper_position": dummy_gripper,
    
    "prompt": "pick up the fork"
}

print("5. 开始推理...")
result = policy.infer(example)
actions = result["actions"]

print("-" * 30)
print("推理成功！")
print(f"输出动作的形状 (Shape): {actions.shape}")
print("-" * 30)
print("输出的前3步动作数据 (Action Chunk):")
print(actions[:3])
print("-" * 30)
