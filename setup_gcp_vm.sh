#!/bin/bash
# setup_gcp_vm.sh — Bootstrap script for GCP Compute Engine
#
# Usage: run this as the VM's startup script, OR SSH in and run it manually
# after uploading stock_recommender.zip to the home directory.
#
# Assumes: Ubuntu 22.04/24.04 image, e2-standard-2 (8GB RAM) or larger.

set -e

echo "=== Updating packages ==="
sudo apt-get update -y

echo "=== Installing Docker + Compose plugin ==="
sudo apt-get install -y docker.io docker-compose-plugin unzip python3-pip
sudo usermod -aG docker "$USER"

echo "=== Docker installed. You may need to log out/in (or run 'newgrp docker') ==="
echo "    for the group membership to take effect before running docker commands."

# If the project zip is already in the home directory, unpack it.
if [ -f "$HOME/stock_recommender.zip" ]; then
    echo "=== Found stock_recommender.zip — unpacking ==="
    cd "$HOME"
    unzip -o stock_recommender.zip
    cd stock_recommender

    echo "=== Installing Python deps (for running ingester.py / train_ml.py / api_access.py outside Docker) ==="
    pip3 install -r requirements.txt --break-system-packages

    echo ""
    echo "=== Setup complete. Next steps (run manually): ==="
    echo "  cd ~/stock_recommender"
    echo "  newgrp docker            # if you weren't re-logged in"
    echo "  docker compose up -d --build"
    echo "  sleep 30                 # let airflow-init finish"
    echo "  python3 ingester.py      # populate real data via yfinance"
    echo "  python3 train_ml.py      # train + log to MLflow"
    echo "  python3 api_access.py    # pull 4 app details via REST APIs"
else
    echo ""
    echo "=== stock_recommender.zip not found in $HOME ==="
    echo "Upload it first, e.g. from your local machine:"
    echo "  gcloud compute scp stock_recommender.zip <instance-name>:~ --zone=<your-zone>"
    echo "Then re-run this script, or unzip + follow README.md manually."
fi
