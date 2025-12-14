"""
Example script demonstrating how to use the Explainability Analysis Tool.

This script shows how to:
1. Load a model
2. Create a dataset
3. Run explainability analysis
4. Generate error reports
"""

import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
import argparse

from explainability_analysis import ExplainabilityAnalyzer, create_sample_dataset

# #region agent log
LOG_PATH = "/Users/tksiankop/Downloads/pokemon_gradcam_starter/.cursor/debug.log"
# #endregion

def log_debug(session_id: str, run_id: str, hypothesis_id: str, location: str, message: str, data: dict):
    """Log debug information to NDJSON file"""
    # #region agent log
    try:
        import json
        import time
        log_entry = {
            "sessionId": session_id,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000)
        }
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass
    # #endregion


class SimplePokemonClassifier(nn.Module):
    """
    A simple CNN for Pokémon classification.
    This is a placeholder - replace with your actual trained model.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_model(
    model_path: str = None, 
    num_classes: int = None,
    model_arch: str = "resnet18",
    device: torch.device = None
) -> nn.Module:
    """
    Load a trained model from a checkpoint file.
    
    Args:
        model_path: Path to the trained model checkpoint (.pth, .pt, or .ckpt)
        num_classes: Number of classes (required if loading from checkpoint)
        model_arch: Model architecture name (resnet18, resnet34, resnet50, efficientnet_b0, etc.)
                   or 'custom' to use SimplePokemonClassifier
        device: Device to load the model on
    
    Returns:
        Loaded model in eval mode
    """
    # #region agent log
    log_debug("debug-session", "run1", "J", "run_explainability.py:load_model",
             "Loading model", {"model_path": model_path, "num_classes": num_classes, 
                            "model_arch": model_arch})
    # #endregion
    
    if model_path and Path(model_path).exists():
        print(f"Loading trained model from: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Determine number of classes from checkpoint if not provided
        if num_classes is None:
            if isinstance(checkpoint, dict):
                # Try to infer from checkpoint
                if 'num_classes' in checkpoint:
                    num_classes = checkpoint['num_classes']
                elif 'model' in checkpoint and hasattr(checkpoint['model'], 'classifier'):
                    # Try to get from model state dict
                    state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
                    if isinstance(state_dict, dict):
                        # Look for classifier layer
                        for key in state_dict.keys():
                            if 'classifier' in key or 'fc' in key:
                                num_classes = state_dict[key].shape[0]
                                break
                else:
                    raise ValueError("Cannot determine num_classes. Please specify it.")
            else:
                raise ValueError("Cannot determine num_classes from checkpoint. Please specify it.")
        
        # Create model architecture
        if model_arch.lower() == 'custom' or model_arch.lower() == 'simple':
            model = SimplePokemonClassifier(num_classes)
        elif model_arch.lower().startswith('resnet'):
            # ResNet architectures
            arch_map = {
                'resnet18': models.resnet18,
                'resnet34': models.resnet34,
                'resnet50': models.resnet50,
                'resnet101': models.resnet101,
                'resnet152': models.resnet152,
            }
            if model_arch.lower() not in arch_map:
                raise ValueError(f"Unsupported ResNet architecture: {model_arch}")
            
            model = arch_map[model_arch.lower()](pretrained=False)
            # Modify final layer for number of classes
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_arch.lower().startswith('efficientnet'):
            try:
                import torchvision.models as tv_models
                # EfficientNet architectures
                arch_map = {
                    'efficientnet_b0': tv_models.efficientnet_b0,
                    'efficientnet_b1': tv_models.efficientnet_b1,
                    'efficientnet_b2': tv_models.efficientnet_b2,
                    'efficientnet_b3': tv_models.efficientnet_b3,
                    'efficientnet_b4': tv_models.efficientnet_b4,
                }
                if model_arch.lower() not in arch_map:
                    raise ValueError(f"Unsupported EfficientNet architecture: {model_arch}")
                
                model = arch_map[model_arch.lower()](pretrained=False)
                # Modify classifier for number of classes
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(model.classifier[1].in_features, num_classes)
                )
            except AttributeError:
                # Older torchvision versions
                raise ValueError(f"EfficientNet not available in this torchvision version. Use ResNet or custom.")
        else:
            raise ValueError(f"Unsupported model architecture: {model_arch}")
        
        # Load state dict
        if isinstance(checkpoint, dict):
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                # Assume the dict itself is the state dict
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (from DataParallel)
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict, strict=False)
        else:
            # Checkpoint is directly the state dict
            model.load_state_dict(checkpoint, strict=False)
        
        model.eval()
        
        # #region agent log
        log_debug("debug-session", "run1", "J", "run_explainability.py:load_model",
                 "Model loaded from checkpoint", {"num_classes": num_classes, "model_arch": model_arch})
        # #endregion
        
        return model
    
    # Fallback: create a random model for demo
    print("Warning: No model path provided. Using random initialized model for demo.")
    if num_classes is None:
        num_classes = 40  # Default based on available images
    
    model = SimplePokemonClassifier(num_classes)
    model.eval()
    
    # #region agent log
    log_debug("debug-session", "run1", "J", "run_explainability.py:load_model",
             "Model created (random)", {"num_classes": num_classes})
    # #endregion
    
    return model


def get_class_names(image_dir: str) -> list:
    """Extract class names from image filenames"""
    image_dir = Path(image_dir)
    class_names = sorted(set(
        img.stem for img in image_dir.glob("*.png")
    ))
    return class_names


def main():
    """Main function to run explainability analysis"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Pokémon Classifier Explainability Analysis")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to trained model checkpoint (.pth, .pt, or .ckpt)")
    parser.add_argument("--model_arch", type=str, default="resnet18",
                       help="Model architecture (resnet18, resnet50, efficientnet_b0, custom)")
    parser.add_argument("--image_dir", type=str, default="data/pokemon_pics",
                       help="Directory containing Pokémon images")
    parser.add_argument("--output_dir", type=str, default="explainability_results",
                       help="Output directory for results")
    parser.add_argument("--max_samples", type=int, default=20,
                       help="Maximum number of samples to analyze")
    
    args = parser.parse_args()
    
    # #region agent log
    log_debug("debug-session", "run1", "K", "run_explainability.py:main",
             "Starting explainability analysis", {"model_path": args.model_path,
                                                 "model_arch": args.model_arch})
    # #endregion
    
    # Configuration
    image_dir = args.image_dir
    output_dir = args.output_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("Pokémon Classifier Explainability Analysis")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Image directory: {image_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Get class names from images
    class_names = get_class_names(image_dir)
    print(f"Found {len(class_names)} classes")
    print(f"Classes: {', '.join(class_names[:10])}{'...' if len(class_names) > 10 else ''}")
    print()
    
    # #region agent log
    log_debug("debug-session", "run1", "K", "run_explainability.py:main",
             "Class names extracted", {"num_classes": len(class_names)})
    # #endregion
    
    # Load model
    print("Loading model...")
    model = load_model(
        model_path=args.model_path,
        num_classes=len(class_names),
        model_arch=args.model_arch,
        device=device
    )
    model.to(device)
    print("Model loaded.")
    print()
    
    # #region agent log
    log_debug("debug-session", "run1", "K", "run_explainability.py:main",
             "Model loaded to device", {"device": str(device)})
    # #endregion
    
    # Create dataset
    print("Creating dataset...")
    dataset = create_sample_dataset(image_dir, class_names)
    print(f"Dataset created: {len(dataset)} samples")
    print()
    
    # #region agent log
    log_debug("debug-session", "run1", "K", "run_explainability.py:main",
             "Dataset created", {"dataset_size": len(dataset)})
    # #endregion
    
    # Initialize analyzer
    print("Initializing explainability analyzer...")
    analyzer = ExplainabilityAnalyzer(
        model=model,
        device=device,
        class_names=class_names,
        output_dir=output_dir
    )
    print("Analyzer initialized.")
    print()
    
    # Run analysis
    print("Running analysis on dataset...")
    print("(This may take a while depending on dataset size)")
    analyzer.analyze_dataset(dataset, max_samples=args.max_samples)
    print(f"Analysis complete!")
    print(f"  - Correct predictions: {len(analyzer.correct_predictions)}")
    print(f"  - Incorrect predictions: {len(analyzer.incorrect_predictions)}")
    print()
    
    # Generate comparison visualizations
    print("Generating comparison visualizations...")
    analyzer.generate_comparison_visualizations(num_examples=5)
    print("Comparisons saved.")
    print()
    
    # Generate error report
    print("Generating error report...")
    report = analyzer.generate_error_report()
    analyzer.save_report(report)
    print()
    
    print("=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}/")
    print("  - correct/: Correct prediction visualizations")
    print("  - incorrect/: Incorrect prediction visualizations")
    print("  - comparisons/: Comparison visualizations")
    print("  - error_report.json: Detailed error report (JSON)")
    print("  - error_report.txt: Human-readable error report")
    print()
    
    # Print summary
    if "summary" in report:
        print("SUMMARY")
        print("-" * 60)
        print(f"Accuracy: {report['summary']['accuracy']:.2%}")
        print(f"Total: {report['summary']['total_predictions']}")
        print(f"Correct: {report['summary']['correct']}")
        print(f"Incorrect: {report['summary']['incorrect']}")
        print()
    
    if "most_common_errors" in report and report["most_common_errors"]:
        print("TOP ERRORS")
        print("-" * 60)
        for error in report["most_common_errors"][:5]:
            print(f"  {error['true_class']} -> {error['predicted_class']} "
                  f"({error['count']} times)")


if __name__ == "__main__":
    main()

