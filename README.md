# Pokémon Classifier with Grad-CAM Explainability

A complete PyTorch implementation for training a Pokémon classifier and analyzing model decisions using Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations.

##  Features

- **Model Training**: Train ResNet or EfficientNet models on Pokémon images
- **Grad-CAM Visualizations**: See where the model focuses when making predictions
- **Error Analysis**: Identify common mistakes and confusion patterns
- **Explainability Reports**: Detailed analysis of correct vs incorrect predictions
- **Pattern Detection**: Analyze similar colors/shapes causing confusion

##  Results

- **Accuracy**: 80% on test set (16/20 correct predictions)
- **Model**: ResNet18 with ImageNet pretrained weights
- **Classes**: 82 Pokémon species
- **Visualizations**: Grad-CAM heatmaps for all predictions

##  Quick Start

### 1. Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision matplotlib opencv-python pillow numpy
```

### 2. Train a Model

```bash
python train_pokemon_classifier.py --epochs 20 --batch_size 16
```

### 3. Run Explainability Analysis

```bash
python run_explainability.py \
    --model_path checkpoints/best.pth \
    --model_arch resnet18 \
    --max_samples 20
```

##  Project Structure

```
pokemon_gradcam_starter/
├── data/
│   └── pokemon_pics/          # Pokémon images (82 classes)
├── checkpoints/               # Saved model checkpoints
├── explainability_results/    # Analysis outputs
│   ├── correct/              # Correct prediction visualizations
│   ├── incorrect/            # Incorrect prediction visualizations
│   ├── comparisons/          # Comparison visualizations
│   └── error_report.txt      # Error analysis report
├── gradcam.py                # Grad-CAM implementation
├── explainability_analysis.py # Main analysis tool
├── train_pokemon_classifier.py # Training script
├── run_explainability.py      # Analysis runner
├── utils_vis.py              # Visualization utilities
└── README.md                 # This file
```

##  Usage Examples

### Training

```bash
# Basic training
python train_pokemon_classifier.py --epochs 20

# With ResNet50
python train_pokemon_classifier.py --model_arch resnet50 --epochs 30

# Custom settings
python train_pokemon_classifier.py \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.0005 \
    --checkpoint_dir my_checkpoints
```

### Explainability Analysis

```bash
# With trained model
python run_explainability.py \
    --model_path checkpoints/best.pth \
    --model_arch resnet18

# Analyze more samples
python run_explainability.py \
    --model_path checkpoints/best.pth \
    --model_arch resnet18 \
    --max_samples 100
```

##  Key Features

### Grad-CAM Visualizations
- Heatmaps showing where the model focuses
- Side-by-side comparisons of correct vs incorrect predictions
- Overlay visualizations combining original image with heatmap

### Error Analysis
- Most common errors identification
- Similar color/shape confusion detection
- Model bias analysis
- Heatmap intensity comparison

### Reports
- JSON format for programmatic access
- Human-readable text reports
- Error pattern insights

## 🎓 Example Results

### Correct Prediction
- **Model focuses on**: Distinctive features (e.g., Bulbasaur's plant bulb)
- **Attention intensity**: Higher (0.39 average)
- **Visualization**: Clear, focused heatmap

### Incorrect Prediction
- **Model focuses on**: Wrong or scattered features
- **Attention intensity**: Lower (0.30 average)
- **Common causes**: Similar colors, similar shapes

##  Technical Details

- **Framework**: PyTorch
- **Architectures**: ResNet18/34/50/101/152, EfficientNet-B0/B1/B2
- **Image Size**: 224x224
- **Normalization**: ImageNet statistics
- **Grad-CAM**: Custom implementation using `torch.autograd.grad`

##  Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision
- matplotlib
- opencv-python
- pillow
- numpy

##  Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

##  License

This project is open source and available for educational purposes.

##  Acknowledgments

- Grad-CAM paper: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
- PyTorch and torchvision teams
- Pokémon images dataset

##  Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This project demonstrates explainable AI techniques for image classification. The model achieves 80% accuracy on the test set and provides detailed visualizations of its decision-making process.
