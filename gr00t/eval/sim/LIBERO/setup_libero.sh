#!/usr/bin/env bash
set -euxo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_REPO="$SCRIPT_DIR/../../../.."
LIBERO_REPO="$PROJECT_REPO/external_dependencies/LIBERO"
LIBERO_REPO_REL="external_dependencies/LIBERO"
LIBERO_UV_ENV="$SCRIPT_DIR/libero_uv"
LIBERO_SETUP_MODE="${LIBERO_SETUP_MODE:-auto}"
LIBERO_CONDA_ENV_NAME="${LIBERO_CONDA_ENV_NAME:-libero}"
LIBERO_INSTALL_ROOT="${LIBERO_INSTALL_ROOT:-/root/autodl-tmp/libero}"
LIBERO_CONDA_PREFIX="${LIBERO_CONDA_PREFIX:-$LIBERO_INSTALL_ROOT/conda-envs/$LIBERO_CONDA_ENV_NAME}"
LIBERO_CONDA_PKGS_DIRS="${LIBERO_CONDA_PKGS_DIRS:-$LIBERO_INSTALL_ROOT/conda-pkgs}"
LIBERO_PIP_CACHE_DIR="${LIBERO_PIP_CACHE_DIR:-$LIBERO_INSTALL_ROOT/pip-cache}"
LIBERO_TMPDIR="${LIBERO_TMPDIR:-$LIBERO_INSTALL_ROOT/tmp}"
LIBERO_CONFIG_ROOT="${LIBERO_CONFIG_PATH:-$LIBERO_INSTALL_ROOT/config}"

if [ -f /etc/network_turbo ]; then
  source /etc/network_turbo
fi

git -C "$PROJECT_REPO" submodule update --init -- "$LIBERO_REPO_REL"

install_python_stack_with_pip() {
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install --requirement "$LIBERO_REPO/requirements.txt"
  python -m pip install -e "$LIBERO_REPO" --config-settings editable_mode=compat
  python -m pip install --editable "$PROJECT_REPO" --no-deps
  python -m pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    pydantic \
    av \
    tianshou==0.5.1 \
    tyro \
    pandas \
    dm-tree \
    einops==0.8.1 \
    albumentations==1.4.18 \
    pyzmq
  python -m pip install \
    transformers==4.51.3 \
    msgpack==1.1.0 \
    msgpack-numpy==0.4.8 \
    gymnasium==0.29.1
  python -m pip install numpy==1.26.4
  python -m pip install --editable "$PROJECT_REPO" --no-deps
}

initialize_libero_config() {
  mkdir -p "$LIBERO_CONFIG_ROOT"
  if [ ! -f "$LIBERO_CONFIG_ROOT/config.yaml" ]; then
    printf 'n\n' | python -c "from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs; register_libero_envs()"
  else
    python -c "from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs; register_libero_envs()"
  fi
}

validate_libero_env() {
  python - <<'PY'
from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
register_libero_envs()
import gymnasium as gym
env = gym.make("libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate")
env.reset()
env.step(env.action_space.sample())
env.close()
print("Env OK:", type(env))
PY
}

link_conda_python_into_libero_uv() {
  mkdir -p "$LIBERO_UV_ENV/.venv/bin"
  cat > "$LIBERO_UV_ENV/.venv/bin/python" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export LIBERO_CONFIG_PATH="$LIBERO_CONFIG_ROOT"
exec "$CONDA_PREFIX/bin/python" "\$@"
EOF
  chmod +x "$LIBERO_UV_ENV/.venv/bin/python"
  ln -sfn python "$LIBERO_UV_ENV/.venv/bin/python3"
  cat > "$LIBERO_UV_ENV/.venv/bin/pip" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec "$CONDA_PREFIX/bin/pip" "\$@"
EOF
  chmod +x "$LIBERO_UV_ENV/.venv/bin/pip"
}

should_use_conda=0
if [ "$LIBERO_SETUP_MODE" = "conda" ]; then
  should_use_conda=1
elif [ "$LIBERO_SETUP_MODE" = "uv" ]; then
  should_use_conda=0
elif ! command -v uv >/dev/null 2>&1; then
  should_use_conda=1
fi

if [ "$should_use_conda" = "1" ]; then
  eval "$(conda shell.bash hook)"
  mkdir -p "$LIBERO_INSTALL_ROOT" "$LIBERO_CONDA_PKGS_DIRS" "$LIBERO_PIP_CACHE_DIR" "$LIBERO_TMPDIR"
  export CONDA_PKGS_DIRS="$LIBERO_CONDA_PKGS_DIRS"
  export PIP_CACHE_DIR="$LIBERO_PIP_CACHE_DIR"
  export TMPDIR="$LIBERO_TMPDIR"
  export LIBERO_CONFIG_PATH="$LIBERO_CONFIG_ROOT"
  if [ ! -d "$LIBERO_CONDA_PREFIX" ]; then
    conda create -y --override-channels \
      -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
      -c defaults \
      --prefix "$LIBERO_CONDA_PREFIX" \
      python=3.10 pip
  fi
  conda activate "$LIBERO_CONDA_PREFIX"
  install_python_stack_with_pip
  link_conda_python_into_libero_uv
  initialize_libero_config
  validate_libero_env
  echo "LIBERO ready in conda env: $LIBERO_CONDA_PREFIX"
  echo "Compat client python: $LIBERO_UV_ENV/.venv/bin/python3"
  exit 0
fi

rm -rf "$LIBERO_UV_ENV"
mkdir -p "$LIBERO_UV_ENV"
uv venv "$LIBERO_UV_ENV/.venv" --python 3.10
source "$LIBERO_UV_ENV/.venv/bin/activate"
uv pip install --requirements "$LIBERO_REPO/requirements.txt"
uv pip install -e "$LIBERO_REPO" --config-settings editable_mode=compat
uv pip install --editable "$PROJECT_REPO" --no-deps
uv pip install torch==2.5.1 torchvision==0.20.1 pydantic av tianshou==0.5.1 tyro pandas dm_tree einops==0.8.1 albumentations==1.4.18 zmq
uv pip install transformers==4.51.3 msgpack==1.1.0 msgpack-numpy==0.4.8 gymnasium==0.29.1
uv pip install numpy==1.26.4
uv pip install --editable "$PROJECT_REPO" --no-deps
initialize_libero_config
validate_libero_env
