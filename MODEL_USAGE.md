# Using Trained Models with Explainability Analysis

## Quick Start

### With a Trained Model

```bash
python run_explainability.py --model_path path/to/your/model.pth --model_arch resnet18
```

### Without a Model (Demo Mode)

```bash
python run_explainability.py
```

## Command Line Arguments

- `--model_path`: Path to your trained model checkpoint (`.pth`, `.pt`, or `.ckpt`)
- `--model_arch`: Model architecture type. Options:
  - `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
  - `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`, `efficientnet_b3`, `efficientnet_b4`
  - `custom` or `simple` (uses SimplePokemonClassifier)
- `--image_dir`: Directory containing Pokémon images (default: `data/pokemon_pics`)
- `--output_dir`: Output directory for results (default: `explainability_results`)
- `--max_samples`: Maximum number of samples to analyze (default: 20)

## Examples

### ResNet18 Model
```bash
python run_explainability.py \
    --model_path checkpoints/pokemon_resnet18.pth \
    --model_arch resnet18 \
    --max_samples 50
```

### ResNet50 Model
```bash
python run_explainability.py \
    --model_path checkpoints/pokemon_resnet50.pth \
    --model_arch resnet50
```

### EfficientNet Model
```bash
python run_explainability.py \
    --model_path checkpoints/pokemon_efficientnet.pth \
    --model_arch efficientnet_b0
```

### Custom Model Architecture
If you have a custom model architecture, you can modify the `load_model` function in `run_explainability.py` to load it:

```python
def load_model(...):
    # ... existing code ...
    if model_arch.lower() == 'your_custom_model':
        from your_model_file import YourCustomModel
        model = YourCustomModel(num_classes=num_classes)
        model.load_state_dict(checkpoint['state_dict'])
        return model
```

## Model Checkpoint Format

The tool supports multiple checkpoint formats:

1. **State dict only**: `torch.save(model.state_dict(), 'model.pth')`
2. **Full checkpoint dict**: 
   ```python
   torch.save({
       'state_dict': model.state_dict(),
       'num_classes': num_classes,
       'epoch': epoch,
       # ... other metadata
   }, 'model.pth')
   ```
3. **Model state dict key**: `checkpoint['model_state_dict']`

The tool will automatically detect and handle DataParallel models (removes `module.` prefix).

## Notes

- The number of classes is automatically inferred from the checkpoint when possible
- If inference fails, you'll need to ensure the number of classes matches your dataset
- The model is automatically set to evaluation mode
- GPU is used automatically if available, otherwise CPU

