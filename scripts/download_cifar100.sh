#!/bin/bash

# Download CIFAR-100 dataset script for Aurora
# This script downloads and unpacks the CIFAR-100 dataset to the data/ directory

set -e  # Exit on any error

# Configuration
DATA_DIR="data"
CIFAR_URL="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR_ARCHIVE="cifar-100-python.tar.gz"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting CIFAR-100 download and setup...${NC}"

# Create data directory if it doesn't exist
if [ ! -d "$DATA_DIR" ]; then
   echo -e "${YELLOW}Creating data directory: $DATA_DIR${NC}"
   mkdir -p "$DATA_DIR"
fi

cd "$DATA_DIR"

# Check if dataset already exists
if [ -d "cifar-100-python" ]; then
   echo -e "${YELLOW}CIFAR-100 dataset already exists in $DATA_DIR/cifar-100-python${NC}"
   echo -e "${GREEN}Dataset setup complete!${NC}"
   exit 0
fi

# Download the dataset
echo -e "${YELLOW}Downloading CIFAR-100 dataset...${NC}"
if command -v wget &> /dev/null; then
   wget "$CIFAR_URL" -O "$CIFAR_ARCHIVE"
elif command -v curl &> /dev/null; then
   curl -L "$CIFAR_URL" -o "$CIFAR_ARCHIVE"
else
   echo -e "${RED}Error: Neither wget nor curl found. Please install one of them.${NC}"
   exit 1
fi

# Verify download
if [ ! -f "$CIFAR_ARCHIVE" ]; then
   echo -e "${RED}Error: Failed to download CIFAR-100 dataset${NC}"
   exit 1
fi

# Extract the archive
echo -e "${YELLOW}Extracting CIFAR-100 dataset...${NC}"
tar -xzf "$CIFAR_ARCHIVE"

# Clean up the archive
echo -e "${YELLOW}Cleaning up archive file...${NC}"
rm "$CIFAR_ARCHIVE"

# Verify extraction
if [ -d "cifar-100-python" ]; then
   echo -e "${GREEN}CIFAR-100 dataset successfully downloaded and extracted!${NC}"
   echo -e "${GREEN}Dataset location: $DATA_DIR/cifar-100-python${NC}"
   echo -e "${YELLOW}Dataset contents:${NC}"
   ls -la cifar-100-python/
else
   echo -e "${RED}Error: Failed to extract CIFAR-100 dataset${NC}"
   exit 1
fi

echo -e "${GREEN}Dataset setup complete!${NC}" 