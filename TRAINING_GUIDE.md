# Pokémon Classifier Training Guide

## Quick Start

Train a Pokémon classifier model:

```bash
python train_pokemon_classifier.py --data_dir data/pokemon_pics --epochs 20
```

## Command Line Arguments

- `--data_dir`: Directory containing Pokémon images (default: `data/pokemon_pics`)
- `--model_arch`: Model architecture (default: `resnet18`)
  - Options: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
  - Or: `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 20)
- `--lr`: Learning rate (default: 0.001)
- `--val_split`: Validation split ratio (default: 0.2)
- `--checkpoint_dir`: Directory to save checkpoints (default: `checkpoints`)
- `--pretrained`: Use pretrained ImageNet weights (default: True)
- `--no_pretrained`: Don't use pretrained weights

## Examples

### Basic Training (ResNet18)
```bash
python train_pokemon_classifier.py --epochs 20
```

### Training with ResNet50
```bash
python train_pokemon_classifier.py --model_arch resnet50 --epochs 30 --batch_size 16
```

### Training from Scratch (No Pretrained Weights)
```bash
python train_pokemon_classifier.py --no_pretrained --epochs 50
```

### Training with Custom Settings
```bash
python train_pokemon_classifier.py \
    --model_arch resnet34 \
    --batch_size 64 \
    --epochs 25 \
    --lr 0.0005 \
    --val_split 0.15 \
    --checkpoint_dir my_checkpoints
```

## Data Requirements

### Current Setup
- Images should be in `data/pokemon_pics/` directory
- Each image should be named `pokemon_name.png` (e.g., `Pikachu.png`)
- The script automatically extracts class names from filenames

### Data Augmentation
The training script includes data augmentation:
- Random cropping
- Random horizontal flipping
- Color jittering (brightness, contrast, saturation, hue)

This helps improve model generalization, especially with limited data.

## Training Process

1. **Dataset Loading**: Images are loaded and split into train/validation sets
2. **Model Creation**: Creates the specified architecture with pretrained weights
3. **Training Loop**: 
   - Trains for specified number of epochs
   - Validates after each epoch
   - Saves best model based on validation accuracy
4. **Checkpoint Saving**: 
   - `best.pth`: Best model (highest validation accuracy)
   - `latest.pth`: Latest model from last epoch

## Using Trained Model

After training, use the model with explainability analysis:

```bash
python run_explainability.py \
    --model_path checkpoints/best.pth \
    --model_arch resnet18
```

## Tips for Better Results

1. **More Data**: If you have multiple images per Pokémon, organize them in subdirectories or use a different dataset structure
2. **More Epochs**: For better accuracy, train for more epochs (50-100)
3. **Learning Rate**: Adjust learning rate if training is unstable
4. **Model Architecture**: Larger models (ResNet50, ResNet101) may perform better but require more memory
5. **Data Augmentation**: Already included, but you can modify augmentation in the script

## Monitoring Training

The script prints:
- Training loss and accuracy after every 10 batches
- Validation loss and accuracy after each epoch
- Best model checkpoint notifications

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` (e.g., `--batch_size 16`)
- Use smaller model (e.g., `resnet18` instead of `resnet50`)

### Low Accuracy
- Train for more epochs
- Use pretrained weights (`--pretrained`)
- Check that images are correctly labeled
- Ensure sufficient training data

### Slow Training
- Use GPU if available (automatically detected)
- Reduce batch size if using CPU
- Use smaller model architecture

## Next Steps

After training:
1. Evaluate the model on test data
2. Use explainability analysis to understand model decisions
3. Fine-tune hyperparameters based on results
4. Consider ensemble methods for better performance

