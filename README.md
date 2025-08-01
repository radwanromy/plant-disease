# Plant Disease Classifier 🌿

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-macOS-lightgrey.svg)

A deep learning application that classifies plant diseases from leaf images using TensorFlow. Optimized for Apple Silicon (M4) Macs with Metal GPU acceleration.

## 🌟 Features

- **Image Classification**: Identifies plant diseases from leaf images
- **Apple Silicon Optimized**: Leverages Metal Performance Shaders for GPU acceleration
- **Multiple Interfaces**: Command-line tool and web interface
- **Transfer Learning**: Uses MobileNetV2 pre-trained model
- **Real-time Prediction**: Fast inference (~5-10ms per image on M4)

## 📁 Project Structure
plant-disease/ ├── src/ # Source code │ ├── train.py # Model training script │ ├── predict.py # Prediction script │ ├── create_proper_model.py # Model creation utility │ └── quick_test.py # Quick test script ├── data/ # Dataset directory (not included) ├── models/ # Trained models (not included) ├── templates/ # Flask web templates ├── .gitignore # Git ignore rules ├── requirements.txt # Python dependencies ├── LICENSE # MIT License └── README.md # This file


## 🚀 Getting Started

### Prerequisites

- macOS 15.5
- Python 3.11 (recommended for TensorFlow compatibility)
- Apple Silicon Mac (M4)
- Git installed

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/radwanromy/plant-disease.git
   cd plant-disease
   
2. **Create virtual environment**
python3.11 -m venv ai-venv<br>
source ai-venv/bin/activate

3. **Install dependencies**
pip install -r requirements.txt

4. **Download dataset (optional for training)**
 Download PlantVillage dataset from Mendeley Data
 Extract to data/ directory
 
 
🎯 Usage
**Training a Model**


#### Create a new model
python3.11 -m venv ai-venv<br>
source ai-venv/bin/activate


#### Train with custom dataset (if downloaded)
python src/train.py
 
 
 ## Making Predictions

### Command Line Interface

#### Predict with custom image
python src/predict.py /path/to/your/image.jpg

# Use default image (place sample_leaf.jpg in project root)
python src/predict.py

<img width="649" height="548" alt="image" src="https://github.com/user-attachments/assets/be56619e-a94d-4060-8c69-3fb2f65085f2" />

 
### Web Interface

#### Start Flask server
python app.py

<img width="1576" height="1372" alt="image" src="https://github.com/user-attachments/assets/f137d1f8-1326-4564-898f-50a35f396179" />


#### Open browser to http://localhost:5001
 
### Quick Test
#### Test with synthetic image
python src/quick_test.py


**Acknowledgments**

 Dataset: PlantVillage Dataset (available on Mendeley Data)
 Framework: TensorFlow and Keras teams
 Inspiration: Various plant disease detection research papers
 Apple: For Metal Performance Shaders and Apple Silicon
