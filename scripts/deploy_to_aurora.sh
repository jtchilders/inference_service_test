#!/bin/bash

# Deployment script for Aurora
# This script sets up the software repository and dataset on Aurora

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="config.yaml"
AURORA_HOST="aurora.alcf.anl.gov"
AURORA_USER="${AURORA_USER:-$(whoami)}"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  Aurora Deployment Script${NC}"
echo -e "${BLUE}================================${NC}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
   echo -e "${RED}Error: Configuration file $CONFIG_FILE not found${NC}"
   echo -e "${YELLOW}Please copy config.yaml.example to config.yaml and update with your settings${NC}"
   exit 1
fi

# Check if we're in a git repository
if [ ! -d ".git" ]; then
   echo -e "${RED}Error: Not in a git repository${NC}"
   echo -e "${YELLOW}This script must be run from the root of the git repository${NC}"
   exit 1
fi

# Get git repository info
REPO_URL=$(git config --get remote.origin.url)
CURRENT_BRANCH=$(git branch --show-current)

echo -e "${GREEN}Repository: $REPO_URL${NC}"
echo -e "${GREEN}Branch: $CURRENT_BRANCH${NC}"

# Function to execute SSH command
execute_ssh() {
   local command="$1"
   echo -e "${YELLOW}Executing: $command${NC}"
   ssh "$AURORA_HOST" "$command"
}

# Function to check if directory exists on Aurora
check_remote_dir() {
   local dir="$1"
   execute_ssh "test -d '$dir' && echo 'exists'"
}

# Get Aurora base directory from config
AURORA_BASE_DIR=$(python3 -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print(config['working_dirs']['aurora_base'])
")

echo -e "${GREEN}Aurora base directory: $AURORA_BASE_DIR${NC}"

# Step 1: Create base directory on Aurora
echo -e "${BLUE}Step 1: Creating base directory on Aurora${NC}"
execute_ssh "mkdir -p $AURORA_BASE_DIR"

# Step 2: Check if repository exists
echo -e "${BLUE}Step 2: Checking repository status${NC}"
REPO_PATH="$AURORA_BASE_DIR/inference_service_test"
REPO_EXISTS=$(check_remote_dir "$REPO_PATH")

if [ "$REPO_EXISTS" = "exists" ]; then
   echo -e "${GREEN}Repository exists, updating...${NC}"
   
   # Update existing repository
   execute_ssh "cd $REPO_PATH && git fetch origin"
   execute_ssh "cd $REPO_PATH && git checkout $CURRENT_BRANCH"
   execute_ssh "cd $REPO_PATH && git pull origin $CURRENT_BRANCH"
   execute_ssh "cd $REPO_PATH && git status"
else
   echo -e "${GREEN}Repository does not exist, cloning...${NC}"
   
   # Clone repository
   execute_ssh "cd $AURORA_BASE_DIR && git clone $REPO_URL inference_service_test"
   execute_ssh "cd $REPO_PATH && git checkout $CURRENT_BRANCH"
fi

# Step 3: Verify repository setup
echo -e "${BLUE}Step 3: Verifying repository setup${NC}"
execute_ssh "ls -la $REPO_PATH"
execute_ssh "cd $REPO_PATH && python --version"

# Step 4: Check and download dataset
echo -e "${BLUE}Step 4: Setting up CIFAR-100 dataset${NC}"
DATA_DIR=$(python3 -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print(config['data']['dir'])
")

AURORA_DATA_DIR="$REPO_PATH/$DATA_DIR"
DATASET_EXISTS=$(check_remote_dir "$AURORA_DATA_DIR/cifar-100-python")

if [ "$DATASET_EXISTS" = "exists" ]; then
   echo -e "${GREEN}CIFAR-100 dataset already exists${NC}"
   execute_ssh "ls -la $AURORA_DATA_DIR/cifar-100-python"
else
   echo -e "${GREEN}Downloading CIFAR-100 dataset...${NC}"
   
   # Create data directory and download dataset
   execute_ssh "mkdir -p $AURORA_DATA_DIR"
   execute_ssh "cd $AURORA_DATA_DIR && wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
   execute_ssh "cd $AURORA_DATA_DIR && tar -xzf cifar-100-python.tar.gz"
   execute_ssh "cd $AURORA_DATA_DIR && rm cifar-100-python.tar.gz"
   execute_ssh "ls -la $AURORA_DATA_DIR/cifar-100-python"
fi

# Step 5: Verify deployment
echo -e "${BLUE}Step 5: Verifying deployment${NC}"

# Check source code structure
execute_ssh "test -d $REPO_PATH/src && echo '✓ Source code directory exists'"
execute_ssh "test -f $REPO_PATH/src/training/train.py && echo '✓ Training script exists'"
execute_ssh "test -d $AURORA_DATA_DIR/cifar-100-python && echo '✓ Dataset exists'"
execute_ssh "test -f $REPO_PATH/requirements.txt && echo '✓ Requirements.txt exists'"

# Check Python environment
echo -e "${GREEN}Python environment on Aurora:${NC}"
execute_ssh "cd $REPO_PATH && python --version"

echo -e "${BLUE}================================${NC}"
echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "${BLUE}================================${NC}"
echo -e "${GREEN}Repository: $REPO_PATH${NC}"
echo -e "${GREEN}Dataset: $AURORA_DATA_DIR/cifar-100-python${NC}"
echo -e "${YELLOW}You can now run the LangGraph workflow${NC}" 