#!/bin/bash

set -euo pipefail

ROOT=$PWD

GENRL_TAG="v0.1.1"

export IDENTITY_PATH
export GENSYN_RESET_CONFIG
export CONNECT_TO_TESTNET=true
export ORG_ID
export HF_HUB_DOWNLOAD_TIMEOUT=120
export SWARM_CONTRACT="0xFaD7C5e93f28257429569B854151A1B8DCD404c2"
export HUGGINGFACE_ACCESS_TOKEN="None"

DEFAULT_IDENTITY_PATH="$ROOT"/swarm.pem
IDENTITY_PATH=${IDENTITY_PATH:-$DEFAULT_IDENTITY_PATH}

DOCKER=${DOCKER:-""}
GENSYN_RESET_CONFIG=${GENSYN_RESET_CONFIG:-""}

if [ -n "$DOCKER" ]; then
    volumes=(
        /home/gensyn/rl_swarm/modal-login/temp-data
        /home/gensyn/rl_swarm/keys
        /home/gensyn/rl_swarm/configs
        /home/gensyn/rl_swarm/logs
    )

    for volume in ${volumes[@]}; do
        sudo chown -R 1001:1001 $volume
    done
fi

CPU_ONLY=${CPU_ONLY:-""}
ORG_ID=${ORG_ID:-""}

GREEN_TEXT="\033[32m"
BLUE_TEXT="\033[34m"
RED_TEXT="\033[31m"
RESET_TEXT="\033[0m"

echo_green() {
    echo -e "$GREEN_TEXT$1$RESET_TEXT"
}

echo_blue() {
    echo -e "$BLUE_TEXT$1$RESET_TEXT"
}

echo_red() {
    echo -e "$RED_TEXT$1$RESET_TEXT"
}

ROOT_DIR="$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)"

cleanup() {
    echo_green ">> Shutting down trainer..."

    kill -- -$$ || true

    exit 0
}

errnotify() {
    echo_red ">> An error was detected while running rl-swarm. See $ROOT/logs for full logs."
}

trap cleanup EXIT
trap errnotify ERR

echo -e "\033[38;5;224m"
cat << "EOF"
    ██████  ██            ███████ ██     ██  █████  ██████  ███    ███
    ██   ██ ██            ██      ██     ██ ██   ██ ██   ██ ████  ████
    ██████  ██      █████ ███████ ██  █  ██ ███████ ██████  ██ ████ ██
    ██   ██ ██                 ██ ██ ███ ██ ██   ██ ██   ██ ██  ██  ██
    ██   ██ ███████       ███████  ███ ███  ██   ██ ██   ██ ██      ██

    From Gensyn

EOF

mkdir -p "$ROOT/logs"

if [ "$CONNECT_TO_TESTNET" = true ]; then
    echo "Please login to create an Ethereum Server Wallet"
    cd modal-login

    if ! command -v node > /dev/null 2>&1; then
        echo "Node.js not found. Installing NVM and latest Node.js..."
        export NVM_DIR="$HOME/.nvm"
        if [ ! -d "$NVM_DIR" ]; then
            curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
        fi
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
        [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
        nvm install node
    else
        echo "Node.js is already installed: $(node -v)"
    fi

    if ! command -v yarn > /dev/null 2>&1; then
        if grep -qi "ubuntu" /etc/os-release 2> /dev/null || uname -r | grep -qi "microsoft"; then
            echo "Detected Ubuntu or WSL Ubuntu. Installing Yarn via apt..."
            curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
            echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
            sudo apt update && sudo apt install -y yarn
        else
            echo "Yarn not found. Installing Yarn globally with npm (no profile edits)…"
            npm install -g --silent yarn
        fi
    fi

    ENV_FILE="$ROOT"/modal-login/.env
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "3s/.*/SMART_CONTRACT_ADDRESS=$SWARM_CONTRACT/" "$ENV_FILE"
    else
        sed -i "3s/.*/SMART_CONTRACT_ADDRESS=$SWARM_CONTRACT/" "$ENV_FILE"
    fi

    if [ -z "$DOCKER" ]; then
        yarn install --immutable
        echo "Building server"
        yarn build > "$ROOT/logs/yarn.log" 2>&1
    fi
    yarn start >> "$ROOT/logs/yarn.log" 2>&1 &

    SERVER_PID=$!
    echo "Started server process: $SERVER_PID"
    sleep 5

    if [ ! -f "$ROOT/modal-login/temp-data/userData.json" ]; then
        echo_green ">> userData.json not found. Starting tunnel..."

        if ! command -v lt > /dev/null 2>&1; then
            echo "Installing localtunnel..."
            npm install -g localtunnel
        fi

        echo_green ">> Starting localtunnel for port 3000..."
        lt --port 3000 &
        LT_PID=$!

        SERVER_IP=$(hostname -I | awk '{print $1}')
        echo "Server IP: $SERVER_IP"

        sleep 3

        echo "Your tunnel is running! Use the link printed above to visit the website."
        echo "If prompted for a password, use your VPS IP."
    else
        echo_green ">> User data found. Skipping localtunnel."
    fi

    cd ..

    echo_green ">> Waiting for modal userData.json to be created..."
    while [ ! -f "modal-login/temp-data/userData.json" ]; do
        sleep 5
    done
    echo "Found userData.json. Proceeding..."

    ORG_ID=$(awk 'BEGIN { FS = "\"" } !/^[ \t]*[{}]/ { print $(NF - 1); exit }' modal-login/temp-data/userData.json)
    echo "Your ORG_ID is set to: $ORG_ID"

    if [ -n "${LT_PID:-}" ]; then
        kill "$LT_PID" || true
        echo_green ">> Localtunnel has been closed."
    fi

    echo "Waiting for API key to become activated..."
    while true; do
        STATUS=$(curl -s "http://localhost:3000/api/get-api-key-status?orgId=$ORG_ID")
        if [[ "$STATUS" == "activated" ]]; then
            echo "API key is activated! Proceeding..."
            break
        else
            echo "Waiting for API key to be activated..."
            sleep 5
        fi
    done
fi

echo_green ">> Getting requirements..."
pip install --upgrade pip

pip install git+https://github.com/abz-coder/genrl-optimized
pip install reasoning-gym>=0.1.20
pip install trl
pip install vllm==0.7.3
pip install bitsandbytes 
pip install hivemind@git+https://github.com/gensyn-ai/hivemind@639c964a8019de63135a2594663b5bec8e5356dd

if [ ! -d "$ROOT/configs" ]; then
    mkdir "$ROOT/configs"
fi

USE_VLLM=${USE_VLLM:-"y"}

if [[ "$USE_VLLM" =~ ^(y|Y|yes|YES)$ ]]; then
    echo_green ">> USE_VLLM=$USE_VLLM detected. Using VLLM config..."
    cp "$ROOT/rgym_exp/config/rg-swarm.yaml" "$ROOT/configs/rg-swarm.yaml"
else
    echo_green ">> USE_VLLM=$USE_VLLM detected. Using non-VLLM config..."
    cp "$ROOT/rgym_exp/config/rg-swarm-no-vllm.yaml" "$ROOT/configs/rg-swarm.yaml"
fi

echo_green ">> Setup complete!"

if [ -n "$DOCKER" ]; then
    sudo chmod -R 0777 /home/gensyn/rl_swarm/configs
fi

echo_green ">> Done!"

HF_TOKEN=${HF_TOKEN:-""}
if [ -n "${HF_TOKEN}" ]; then
    HUGGINGFACE_ACCESS_TOKEN=${HF_TOKEN}
else
    yn="N"
    HUGGINGFACE_ACCESS_TOKEN="None"
    echo_green ">> Automatically set: NO models will be pushed to Hugging Face Hub"
fi

MODEL_NAME="dnotitia/Smoothie-Qwen3-1.7B"
echo_green ">> Automatically set model: $MODEL_NAME"

if [ -n "$MODEL_NAME" ]; then
    export MODEL_NAME
    echo_green ">> Using model: $MODEL_NAME"
else
    echo_green ">> Using default model from config"
fi

echo_green ">> Good luck in the swarm!"
echo_blue ">> And remember to star the repo on GitHub! --> https://github.com/gensyn-ai/rl-swarm"

python -m rgym_exp.runner.swarm_launcher \
    --config-path "$ROOT/rgym_exp/config" \
    --config-name "rg-swarm.yaml" 

wait
