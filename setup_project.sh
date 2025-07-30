#!/bin/bash

# Create project structure
mkdir -p ~/ai-projects/plant-disease/{data,models,src}
cd ~/ai-projects/plant-disease

# Download dataset (if not exists)
if [ ! -d "data/Plant_leave_diseases_dataset_without_augmentation" ]; then
    echo "Downloading dataset..."
    curl -L -o plant-disease.zip "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded"
    unzip plant-disease.zip -d data/
    rm plant-disease.zip
fi

# Install dependencies
pip install matplotlib pillow seaborn

echo "Setup complete! Run 'python src/train.py' to start training."
