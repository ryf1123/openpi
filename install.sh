#!/bin/bash
# set -e  # 遇到错误立即退出

echo "=========================================="
echo "OpenPI 环境安装脚本 (PyTorch 版本)"
echo "=========================================="

# 1. 克隆仓库
echo "Step 1: 克隆仓库..."
# git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
git clone --recurse-submodules https://github.com/ryf1123/openpi.git
cd openpi

# 2. 创建 Python 3.11+ 虚拟环境
echo "Step 2: 检查并创建 Python 3.11+ 虚拟环境..."
# 确保安装了 Python 3.11
if ! uv python list | grep -q "3.11"; then
    echo "正在安装 Python 3.11..."
    uv python install 3.11
fi

# 删除旧的虚拟环境（如果存在）并创建新的
if [ -d ".venv" ]; then
    echo "删除旧的虚拟环境..."
    rm -rf .venv
fi

echo "正在创建 Python 3.11 虚拟环境..."
uv venv --python 3.11

# 激活虚拟环境
echo "激活虚拟环境..."
source .venv/bin/activate

# 验证 Python 版本
CURRENT_PYTHON_VERSION=$(python --version | awk '{print $2}')
echo "当前使用 Python 版本: $CURRENT_PYTHON_VERSION"

# 3. 安装 Python 依赖
echo "Step 3: 安装 Python 依赖..."
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . 

# 4. 检查 transformers 版本
echo "Step 4: 检查 transformers 版本..."
TRANSFORMERS_VERSION=$(uv pip show transformers | grep "Version:" | awk '{print $2}')
if [ "$TRANSFORMERS_VERSION" != "4.53.2" ]; then
    echo "错误: transformers 版本不正确 (当前: $TRANSFORMERS_VERSION, 需要: 4.53.2)"
    exit 1
fi
echo "✅ transformers 版本正确: $TRANSFORMERS_VERSION"

# 5. 应用 transformers 补丁（自动检测 Python 版本）
echo "Step 5: 应用 transformers 补丁..."
PYTHON_VERSION=$(uv run python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
SITE_PACKAGES=$(uv run python -c "import site; print(site.getsitepackages()[0])")
echo "检测到 Python $PYTHON_VERSION"
echo "Site-packages 路径: $SITE_PACKAGES"

cp -r ./src/openpi/models_pytorch/transformers_replace/* "$SITE_PACKAGES/transformers/"
echo "✅ transformers 补丁已应用"

# run it to download the checkpoints
uv run run-test.py

# 6. 下载并转换模型（如果尚未存在）
echo "Step 6: 转换 JAX 模型到 PyTorch..."
PYTORCH_CHECKPOINT="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch"

if [ -f "$PYTORCH_CHECKPOINT/model.safetensors" ]; then
    echo "⏭️  PyTorch 模型已存在，跳过转换"
else
    echo "正在下载 JAX 模型并转换为 PyTorch..."
    uv run examples/convert_jax_model_to_pytorch.py \
        --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
        --config-name pi05_droid \
        --output-path "$PYTORCH_CHECKPOINT"

fi

# 7. 确保 assets 文件夹存在（包含 norm_stats.json）
echo "Step 7: 确保 assets 文件夹存在..."
if [ ! -d "$PYTORCH_CHECKPOINT/assets" ]; then
    echo "复制 assets 文件夹 (包含 norm_stats.json)..."
    cp -r ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets "$PYTORCH_CHECKPOINT/"
    echo "✅ assets 文件夹已复制"
else
    echo "✅ assets 文件夹已存在"
fi

# run it to download the checkpoints
uv run run-test.py --pytorch
    
echo ""
echo "=========================================="
echo "✅ 安装完成！"
echo "=========================================="
echo ""
echo "现在可以运行："
echo "  uv run run-test.py"
echo ""
echo "或者进入 Python 调试："
echo "  uv run python run-test.py"
echo ""

