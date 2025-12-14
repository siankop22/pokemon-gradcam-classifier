"""
Explainability Analysis Tool for Pokémon Classifier

This tool generates Grad-CAM visualizations for correct and incorrect predictions,
compares them, and generates an error analysis report.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2

from gradcam import GradCAM, get_last_conv_layer
from utils_vis import denormalize, MEAN, STD

# #region agent log
LOG_PATH = "/Users/tksiankop/Downloads/pokemon_gradcam_starter/.cursor/debug.log"
# #endregion

def log_debug(session_id: str, run_id: str, hypothesis_id: str, location: str, message: str, data: dict):
    """Log debug information to NDJSON file"""
    # #region agent log
    try:
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


class ExplainabilityAnalyzer:
    """
    Analyzes model predictions using Grad-CAM and generates error reports.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: List[str],
        output_dir: str = "explainability_results",
        mean: List[float] = MEAN,
        std: List[float] = STD
    ):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.mean = mean
        self.std = std
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "correct").mkdir(exist_ok=True)
        (self.output_dir / "incorrect").mkdir(exist_ok=True)
        (self.output_dir / "comparisons").mkdir(exist_ok=True)
        
        # Setup Grad-CAM
        self.model.eval()
        target_layer = get_last_conv_layer(model)
        self.gradcam = GradCAM(model, target_layer)
        
        # Transform for images
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        
        # Storage for analysis
        self.predictions = []
        self.correct_predictions = []
        self.incorrect_predictions = []
        
        log_debug("debug-session", "init", "A", "explainability_analysis.py:__init__", 
                 "ExplainabilityAnalyzer initialized", 
                 {"num_classes": len(class_names), "output_dir": str(output_dir)})
    
    def load_image(self, image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        """Load and preprocess an image"""
        # #region agent log
        log_debug("debug-session", "run1", "A", "explainability_analysis.py:load_image",
                 "Loading image", {"path": image_path})
        # #endregion
        
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, img
    
    def predict_and_visualize(
        self,
        image_path: str,
        true_label: Optional[int] = None,
        save_heatmap: bool = True
    ) -> Dict:
        """
        Run prediction and generate Grad-CAM visualization.
        
        Returns:
            Dictionary with prediction results and paths to saved visualizations
        """
        # #region agent log
        log_debug("debug-session", "run1", "B", "explainability_analysis.py:predict_and_visualize",
                 "Starting prediction", {"image_path": image_path, "true_label": true_label})
        # #endregion
        
        img_tensor, img = self.load_image(image_path)
        img_batch = img_tensor.unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(img_batch)
            probs = torch.softmax(output, dim=1)
            pred_class = int(output.argmax(dim=1).item())
            confidence = float(probs[0, pred_class].item())
        
        # #region agent log
        log_debug("debug-session", "run1", "B", "explainability_analysis.py:predict_and_visualize",
                 "Prediction made", {"pred_class": pred_class, "confidence": confidence, 
                                   "true_label": true_label})
        # #endregion
        
        # Generate Grad-CAM
        cam = self.gradcam(img_batch, target_class=pred_class)
        
        # #region agent log
        log_debug("debug-session", "run1", "C", "explainability_analysis.py:predict_and_visualize",
                 "Grad-CAM generated", {"cam_shape": cam.shape, "cam_min": float(cam.min()), 
                                       "cam_max": float(cam.max())})
        # #endregion
        
        is_correct = (true_label is None) or (pred_class == true_label)
        
        result = {
            "image_path": image_path,
            "predicted_class": pred_class,
            "predicted_name": self.class_names[pred_class],
            "true_label": true_label,
            "true_name": self.class_names[true_label] if true_label is not None else None,
            "confidence": confidence,
            "is_correct": is_correct,
            "cam": cam,
            "img_tensor": img_tensor,
            "img": img
        }
        
        if save_heatmap:
            # Save visualization
            category = "correct" if is_correct else "incorrect"
            filename = Path(image_path).stem
            save_path = self._save_visualization(
                img_tensor, cam, result, category, filename
            )
            result["visualization_path"] = save_path
        
        # #region agent log
        log_debug("debug-session", "run1", "D", "explainability_analysis.py:predict_and_visualize",
                 "Visualization saved", {"is_correct": is_correct, "save_path": result.get("visualization_path", "N/A")})
        # #endregion
        
        return result
    
    def _save_visualization(
        self,
        img_tensor: torch.Tensor,
        cam: np.ndarray,
        result: Dict,
        category: str,
        filename: str
    ) -> str:
        """Save a triplet visualization (original, heatmap, overlay)"""
        img = denormalize(img_tensor, mean=self.mean, std=self.std).permute(1, 2, 0).cpu().numpy()
        H, W, _ = img.shape
        
        cam_resized = cv2.resize(cam, (W, H))
        heatmap = cv2.applyColorMap((cam_resized * 255).astype("uint8"), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        overlay = 0.5 * heatmap + 0.5 * img
        overlay = np.clip(overlay, 0.0, 1.0)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(img)
        axes[0].axis("off")
        axes[0].set_title("Original")
        
        axes[1].imshow(heatmap)
        axes[1].axis("off")
        axes[1].set_title("Grad-CAM Heatmap")
        
        axes[2].imshow(overlay)
        axes[2].axis("off")
        axes[2].set_title("Overlay")
        
        title = f"{result['predicted_name']}"
        if result['true_label'] is not None:
            title += f" (True: {result['true_name']})"
        title += f" | Conf: {result['confidence']:.2f}"
        
        fig.suptitle(title, fontsize=10)
        plt.tight_layout()
        
        save_path = self.output_dir / category / f"{filename}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def analyze_dataset(
        self,
        dataset: List[Tuple[str, int]],
        max_samples: Optional[int] = None
    ):
        """
        Analyze a dataset of (image_path, label) tuples.
        
        Args:
            dataset: List of (image_path, true_label) tuples
            max_samples: Maximum number of samples to analyze (None for all)
        """
        # #region agent log
        log_debug("debug-session", "run1", "E", "explainability_analysis.py:analyze_dataset",
                 "Starting dataset analysis", {"dataset_size": len(dataset), 
                                             "max_samples": max_samples})
        # #endregion
        
        if max_samples:
            dataset = dataset[:max_samples]
        
        for idx, (image_path, true_label) in enumerate(dataset):
            # #region agent log
            log_debug("debug-session", "run1", "E", "explainability_analysis.py:analyze_dataset",
                     "Processing sample", {"idx": idx, "image_path": image_path, 
                                         "true_label": true_label})
            # #endregion
            
            try:
                result = self.predict_and_visualize(image_path, true_label)
                self.predictions.append(result)
                
                if result["is_correct"]:
                    self.correct_predictions.append(result)
                else:
                    self.incorrect_predictions.append(result)
            except Exception as e:
                # #region agent log
                log_debug("debug-session", "run1", "F", "explainability_analysis.py:analyze_dataset",
                         "Error processing sample", {"idx": idx, "error": str(e)})
                # #endregion
                print(f"Error processing {image_path}: {e}")
                continue
        
        # #region agent log
        log_debug("debug-session", "run1", "G", "explainability_analysis.py:analyze_dataset",
                 "Dataset analysis complete", {"total": len(self.predictions),
                                            "correct": len(self.correct_predictions),
                                            "incorrect": len(self.incorrect_predictions)})
        # #endregion
    
    def generate_comparison_visualizations(self, num_examples: int = 5):
        """Generate side-by-side comparisons of correct vs incorrect predictions"""
        # #region agent log
        log_debug("debug-session", "run1", "H", "explainability_analysis.py:generate_comparison_visualizations",
                 "Generating comparisons", {"num_examples": num_examples})
        # #endregion
        
        num_correct = min(num_examples, len(self.correct_predictions))
        num_incorrect = min(num_examples, len(self.incorrect_predictions))
        
        # Handle empty case
        if num_correct == 0 and num_incorrect == 0:
            print("No predictions to visualize. Skipping comparison generation.")
            # #region agent log
            log_debug("debug-session", "run1", "H", "explainability_analysis.py:generate_comparison_visualizations",
                     "No predictions to visualize", {})
            # #endregion
            return
        
        # Sample random examples
        if num_correct > 0:
            correct_samples = np.random.choice(
                len(self.correct_predictions), num_correct, replace=False
            )
        else:
            correct_samples = []
        
        if num_incorrect > 0:
            incorrect_samples = np.random.choice(
                len(self.incorrect_predictions), num_incorrect, replace=False
            )
        else:
            incorrect_samples = []
        
        # Create comparison figure
        max_cols = max(num_correct, num_incorrect, 1)  # Ensure at least 1 column
        fig, axes = plt.subplots(2, max_cols, 
                                figsize=(4 * max_cols, 8))
        if max_cols == 1:
            axes = axes.reshape(2, 1)
        
        # Plot correct predictions
        for i, idx in enumerate(correct_samples):
            result = self.correct_predictions[idx]
            self._plot_single_comparison(axes[0, i], result, "Correct")
        
        # Plot incorrect predictions
        for i, idx in enumerate(incorrect_samples):
            result = self.incorrect_predictions[idx]
            self._plot_single_comparison(axes[1, i], result, "Incorrect")
        
        # Hide empty subplots
        max_cols = max(num_correct, num_incorrect, 1)
        for i in range(num_correct, max_cols):
            axes[0, i].axis("off")
        for i in range(num_incorrect, max_cols):
            axes[1, i].axis("off")
        
        plt.suptitle("Correct vs Incorrect Predictions Comparison", fontsize=14)
        plt.tight_layout()
        
        save_path = self.output_dir / "comparisons" / "correct_vs_incorrect.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # #region agent log
        log_debug("debug-session", "run1", "H", "explainability_analysis.py:generate_comparison_visualizations",
                 "Comparison saved", {"save_path": str(save_path)})
        # #endregion
    
    def _plot_single_comparison(self, ax, result: Dict, label: str):
        """Plot a single comparison (original + overlay)"""
        img = denormalize(result["img_tensor"], mean=self.mean, std=self.std).permute(1, 2, 0).cpu().numpy()
        cam = result["cam"]
        H, W, _ = img.shape
        
        cam_resized = cv2.resize(cam, (W, H))
        heatmap = cv2.applyColorMap((cam_resized * 255).astype("uint8"), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.5 * heatmap + 0.5 * img
        overlay = np.clip(overlay, 0.0, 1.0)
        
        ax.imshow(overlay)
        ax.axis("off")
        title = f"{result['predicted_name']}"
        if result['true_name']:
            title += f"\n(True: {result['true_name']})"
        ax.set_title(title, fontsize=8)
    
    def generate_error_report(self) -> Dict:
        """
        Generate a comprehensive error analysis report.
        Analyzes common mistakes, similar colors/shapes, confusion patterns.
        """
        # #region agent log
        log_debug("debug-session", "run1", "I", "explainability_analysis.py:generate_error_report",
                 "Generating error report", {"num_errors": len(self.incorrect_predictions)})
        # #endregion
        
        if not self.incorrect_predictions:
            return {"message": "No errors to analyze"}
        
        # Confusion matrix
        confusion = defaultdict(int)
        for result in self.incorrect_predictions:
            true_name = result["true_name"]
            pred_name = result["predicted_name"]
            confusion[(true_name, pred_name)] += 1
        
        # Most common errors
        most_common_errors = sorted(
            confusion.items(), key=lambda x: x[1], reverse=True
        )[:10]
        
        # Class-level error rates
        class_errors = defaultdict(int)
        class_total = defaultdict(int)
        for result in self.predictions:
            true_name = result["true_name"]
            class_total[true_name] += 1
            if not result["is_correct"]:
                class_errors[true_name] += 1
        
        error_rates = {
            name: class_errors[name] / class_total[name] 
            for name in class_total
        }
        worst_classes = sorted(error_rates.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Analyze heatmap patterns (simple: average intensity in different regions)
        correct_heatmap_intensity = []
        incorrect_heatmap_intensity = []
        
        for result in self.correct_predictions[:20]:  # Sample
            correct_heatmap_intensity.append(result["cam"].mean())
        
        for result in self.incorrect_predictions[:20]:  # Sample
            incorrect_heatmap_intensity.append(result["cam"].mean())
        
        avg_correct_intensity = np.mean(correct_heatmap_intensity) if correct_heatmap_intensity else 0
        avg_incorrect_intensity = np.mean(incorrect_heatmap_intensity) if incorrect_heatmap_intensity else 0
        
        # Analyze error patterns: similar colors/shapes
        error_patterns = self._analyze_error_patterns(most_common_errors)
        
        report = {
            "summary": {
                "total_predictions": len(self.predictions),
                "correct": len(self.correct_predictions),
                "incorrect": len(self.incorrect_predictions),
                "accuracy": len(self.correct_predictions) / len(self.predictions) if self.predictions else 0
            },
            "most_common_errors": [
                {
                    "true_class": true,
                    "predicted_class": pred,
                    "count": count
                }
                for (true, pred), count in most_common_errors
            ],
            "worst_performing_classes": [
                {
                    "class_name": name,
                    "error_rate": rate
                }
                for name, rate in worst_classes
            ],
            "heatmap_analysis": {
                "avg_correct_intensity": float(avg_correct_intensity),
                "avg_incorrect_intensity": float(avg_incorrect_intensity),
                "note": "Higher intensity may indicate more focused attention"
            },
            "error_patterns": error_patterns
        }
        
        # #region agent log
        log_debug("debug-session", "run1", "I", "explainability_analysis.py:generate_error_report",
                 "Error report generated", {"num_common_errors": len(most_common_errors),
                                          "num_worst_classes": len(worst_classes),
                                          "num_patterns": len(error_patterns)})
        # #endregion
        
        return report
    
    def _analyze_error_patterns(self, most_common_errors: List) -> Dict:
        """
        Analyze error patterns to identify common mistakes based on similar colors/shapes.
        """
        # #region agent log
        log_debug("debug-session", "run1", "L", "explainability_analysis.py:_analyze_error_patterns",
                 "Analyzing error patterns", {"num_errors": len(most_common_errors)})
        # #endregion
        
        patterns = {
            "similar_colors": [],
            "similar_shapes": [],
            "common_confusions": [],
            "insights": []
        }
        
        # Known Pokémon characteristics for pattern matching
        # This is a simplified heuristic - in practice, you'd analyze actual images
        pokemon_characteristics = self._get_pokemon_characteristics()
        
        # Analyze each error pair
        for (true_name, pred_name), count in most_common_errors[:10]:
            true_char = pokemon_characteristics.get(true_name.lower(), {})
            pred_char = pokemon_characteristics.get(pred_name.lower(), {})
            
            # Check for similar colors
            true_colors = true_char.get("colors", [])
            pred_colors = pred_char.get("colors", [])
            color_overlap = set(true_colors) & set(pred_colors)
            
            if color_overlap:
                patterns["similar_colors"].append({
                    "true": true_name,
                    "predicted": pred_name,
                    "common_colors": list(color_overlap),
                    "count": count
                })
            
            # Check for similar shapes/types
            true_shape = true_char.get("shape", "")
            pred_shape = pred_char.get("shape", "")
            true_type = true_char.get("type", "")
            pred_type = pred_char.get("type", "")
            
            if true_shape and pred_shape and true_shape == pred_shape:
                patterns["similar_shapes"].append({
                    "true": true_name,
                    "predicted": pred_name,
                    "shape": true_shape,
                    "count": count
                })
            
            if true_type and pred_type and true_type == pred_type:
                patterns["common_confusions"].append({
                    "true": true_name,
                    "predicted": pred_name,
                    "type": true_type,
                    "count": count,
                    "reason": f"Both are {true_type}-type Pokémon"
                })
        
        # Generate insights
        insights = self._generate_insights(patterns, most_common_errors)
        patterns["insights"] = insights
        
        # #region agent log
        log_debug("debug-session", "run1", "L", "explainability_analysis.py:_analyze_error_patterns",
                 "Patterns analyzed", {"similar_colors": len(patterns["similar_colors"]),
                                     "similar_shapes": len(patterns["similar_shapes"]),
                                     "common_confusions": len(patterns["common_confusions"])})
        # #endregion
        
        return patterns
    
    def _get_pokemon_characteristics(self) -> Dict:
        """
        Get simplified characteristics of Pokémon for pattern analysis.
        This is a heuristic-based approach. For more accuracy, analyze actual images.
        """
        # Common color/shape patterns in Pokémon
        characteristics = {
            # Similar small round Pokémon
            "cleffa": {"colors": ["pink", "white"], "shape": "round", "type": "fairy"},
            "jigglypuff": {"colors": ["pink", "white"], "shape": "round", "type": "fairy"},
            "pichu": {"colors": ["yellow", "black"], "shape": "round", "type": "electric"},
            
            # Similar quadruped mammals
            "eevee": {"colors": ["brown", "cream"], "shape": "quadruped", "type": "normal"},
            "zigzagoon": {"colors": ["brown", "white", "black"], "shape": "quadruped", "type": "normal"},
            "linoone": {"colors": ["white", "brown"], "shape": "quadruped", "type": "normal"},
            "yamper": {"colors": ["yellow", "brown"], "shape": "quadruped", "type": "electric"},
            
            # Similar bird Pokémon
            "talonflame": {"colors": ["red", "orange", "yellow"], "shape": "bird", "type": "fire"},
            "pidgey": {"colors": ["brown", "cream"], "shape": "bird", "type": "normal"},
            
            # Similar water Pokémon
            "starmie": {"colors": ["purple", "red"], "shape": "star", "type": "water"},
            "staryu": {"colors": ["brown", "yellow"], "shape": "star", "type": "water"},
            "vaporeon": {"colors": ["blue", "white"], "shape": "quadruped", "type": "water"},
            
            # Similar rock/ground types
            "nosepass": {"colors": ["blue", "gray"], "shape": "humanoid", "type": "rock"},
            "probopass": {"colors": ["red", "gray"], "shape": "humanoid", "type": "rock"},
            
            # Similar fire types
            "litten": {"colors": ["red", "black"], "shape": "quadruped", "type": "fire"},
            "pyroar": {"colors": ["orange", "yellow", "brown"], "shape": "quadruped", "type": "fire"},
            
            # Similar grass types
            "bulbasaur": {"colors": ["green", "blue"], "shape": "quadruped", "type": "grass"},
            "ivysaur": {"colors": ["green", "blue"], "shape": "quadruped", "type": "grass"},
            "venusaur": {"colors": ["green", "blue"], "shape": "quadruped", "type": "grass"},
            
            # Similar dragon types
            "dratini": {"colors": ["blue", "white"], "shape": "serpent", "type": "dragon"},
            "dragonair": {"colors": ["blue", "white"], "shape": "serpent", "type": "dragon"},
            
            # Similar evolution lines
            "pawmo": {"colors": ["yellow", "black"], "shape": "quadruped", "type": "electric"},
            "pawmot": {"colors": ["yellow", "black"], "shape": "biped", "type": "electric"},
        }
        
        # Add lowercase versions for case-insensitive matching
        characteristics_lower = {}
        for key, value in characteristics.items():
            characteristics_lower[key.lower()] = value
        
        return characteristics_lower
    
    def _generate_insights(self, patterns: Dict, most_common_errors: List) -> List[str]:
        """
        Generate human-readable insights about error patterns.
        """
        insights = []
        
        # Analyze color-based confusions
        if patterns["similar_colors"]:
            color_groups = defaultdict(list)
            for item in patterns["similar_colors"]:
                for color in item["common_colors"]:
                    color_groups[color].append(f"{item['true']} → {item['predicted']}")
            
            for color, pairs in color_groups.items():
                if len(pairs) >= 2:
                    insights.append(
                        f"Color confusion ({color}): Multiple Pokémon with {color} coloring are being confused. "
                        f"Examples: {', '.join(pairs[:3])}"
                    )
        
        # Analyze shape-based confusions
        if patterns["similar_shapes"]:
            shape_groups = defaultdict(list)
            for item in patterns["similar_shapes"]:
                shape_groups[item["shape"]].append(f"{item['true']} → {item['predicted']}")
            
            for shape, pairs in shape_groups.items():
                if len(pairs) >= 2:
                    insights.append(
                        f"Shape confusion ({shape}): Pokémon with similar {shape} shapes are being confused. "
                        f"Examples: {', '.join(pairs[:3])}"
                    )
        
        # Analyze type-based confusions
        if patterns["common_confusions"]:
            type_groups = defaultdict(list)
            for item in patterns["common_confusions"]:
                type_groups[item["type"]].append(f"{item['true']} → {item['predicted']}")
            
            for ptype, pairs in type_groups.items():
                if len(pairs) >= 2:
                    insights.append(
                        f"Type confusion ({ptype}): {ptype}-type Pokémon are frequently confused with each other. "
                        f"Examples: {', '.join(pairs[:3])}"
                    )
        
        # General insights
        if most_common_errors:
            # Find most confused Pokémon (appears as prediction multiple times)
            prediction_counts = defaultdict(int)
            for (true, pred), count in most_common_errors:
                prediction_counts[pred] += count
            
            most_confused = sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            if most_confused:
                confused_list = [f"{name} ({count} times)" for name, count in most_confused]
                insights.append(
                    f"Most frequently predicted (often incorrectly): {', '.join(confused_list)}. "
                    f"This suggests the model may be biased toward these classes."
                )
        
        # If no specific patterns found, provide general guidance
        if not insights:
            insights.append(
                "Error patterns are diverse. Consider analyzing Grad-CAM heatmaps to understand "
                "what visual features the model is focusing on for each confusion."
            )
        
        return insights
    
    def save_report(self, report: Dict):
        """Save error report to JSON file"""
        report_path = self.output_dir / "error_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        # Also create a human-readable text report
        text_report_path = self.output_dir / "error_report.txt"
        with open(text_report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("Pokémon Classifier Error Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            if "summary" in report:
                f.write("SUMMARY\n")
                f.write("-" * 60 + "\n")
                f.write(f"Total Predictions: {report['summary']['total_predictions']}\n")
                f.write(f"Correct: {report['summary']['correct']}\n")
                f.write(f"Incorrect: {report['summary']['incorrect']}\n")
                f.write(f"Accuracy: {report['summary']['accuracy']:.2%}\n\n")
            else:
                f.write("SUMMARY\n")
                f.write("-" * 60 + "\n")
                f.write("No predictions to analyze.\n\n")
            
            f.write("MOST COMMON ERRORS\n")
            f.write("-" * 60 + "\n")
            for error in report['most_common_errors']:
                f.write(f"True: {error['true_class']:20s} -> Predicted: {error['predicted_class']:20s} "
                       f"(Count: {error['count']})\n")
            f.write("\n")
            
            f.write("WORST PERFORMING CLASSES\n")
            f.write("-" * 60 + "\n")
            for cls in report['worst_performing_classes']:
                f.write(f"{cls['class_name']:20s}: {cls['error_rate']:.2%} error rate\n")
            f.write("\n")
            
            f.write("HEATMAP ANALYSIS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Average Correct Prediction Intensity: {report['heatmap_analysis']['avg_correct_intensity']:.4f}\n")
            f.write(f"Average Incorrect Prediction Intensity: {report['heatmap_analysis']['avg_incorrect_intensity']:.4f}\n")
            f.write(f"Note: {report['heatmap_analysis']['note']}\n")
            f.write("\n")
            
            # Error Pattern Analysis
            if "error_patterns" in report and report["error_patterns"]:
                f.write("ERROR PATTERN ANALYSIS\n")
                f.write("-" * 60 + "\n")
                
                patterns = report["error_patterns"]
                
                if patterns.get("similar_colors"):
                    f.write("\nSimilar Colors Confusion:\n")
                    for item in patterns["similar_colors"][:5]:
                        f.write(f"  {item['true']} → {item['predicted']}: "
                               f"Common colors: {', '.join(item['common_colors'])} "
                               f"({item['count']} times)\n")
                
                if patterns.get("similar_shapes"):
                    f.write("\nSimilar Shapes Confusion:\n")
                    for item in patterns["similar_shapes"][:5]:
                        f.write(f"  {item['true']} → {item['predicted']}: "
                               f"Both are {item['shape']}-shaped ({item['count']} times)\n")
                
                if patterns.get("common_confusions"):
                    f.write("\nType-Based Confusion:\n")
                    for item in patterns["common_confusions"][:5]:
                        f.write(f"  {item['true']} → {item['predicted']}: {item['reason']} "
                               f"({item['count']} times)\n")
                
                if patterns.get("insights"):
                    f.write("\nKey Insights:\n")
                    for i, insight in enumerate(patterns["insights"], 1):
                        f.write(f"  {i}. {insight}\n")
        
        print(f"Reports saved to:")
        print(f"  - {report_path}")
        print(f"  - {text_report_path}")


def create_sample_dataset(image_dir: str, class_names: List[str]) -> List[Tuple[str, int]]:
    """
    Create a sample dataset from image directory.
    Assumes images are named like 'pokemon_name.png' and maps to class_names.
    """
    image_dir = Path(image_dir)
    dataset = []
    
    for img_path in image_dir.glob("*.png"):
        # Extract Pokemon name from filename (remove extension, handle case)
        pokemon_name = img_path.stem
        # Try to find matching class (case-insensitive)
        try:
            class_idx = next(
                i for i, name in enumerate(class_names) 
                if name.lower() == pokemon_name.lower()
            )
            dataset.append((str(img_path), class_idx))
        except StopIteration:
            # If not found, skip or assign to a default class
            continue
    
    return dataset

