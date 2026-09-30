# ============================================================================
# FILE: utils/visualization_export.py
# Export and reload structured visualization data from evaluation outputs.
#
# Purpose:
# - Saves plotting-ready JSON files derived from evaluation results.
#
# Workflow:
# - Writes confusion matrices, ROC curves, PR curves, flow diagrams, and
#   summary statistics into a visualization directory.
#
# Use this file when:
# - You want portable structured plotting artifacts outside the main figure
#   generation path.
# ============================================================================
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def export_visualization_data(results: Dict, output_dir: Path):
    """
    Export visualization data from evaluation results
    
    Args:
        results: Results dictionary from evaluation
        output_dir: Directory to save visualization data
    
    Creates files:
        - confusion_matrices.json
        - roc_curves.json
        - pr_curves.json
        - flow_diagram.json
        - layer_stats.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrices
    if "metrics" in results:
        cm_data = {
            "overall": results["metrics"]["confusion_matrix"]
        }
        
        # Per-layer CMs
        if "per_layer_metrics" in results:
            for layer_name, metrics in results["per_layer_metrics"].items():
                cm_data[layer_name] = metrics["confusion_matrix"]
        
        with open(output_dir / "confusion_matrices.json", 'w') as f:
            json.dump(cm_data, f, indent=2)
        
        print(f"   ✅ Saved confusion matrices")
    
    # 2. ROC Curves
    if "metrics" in results and results["metrics"].get("roc_curve"):
        roc_data = {
            "overall": results["metrics"]["roc_curve"]
        }
        
        # Per-layer ROCs
        if "per_layer_metrics" in results:
            for layer_name, metrics in results["per_layer_metrics"].items():
                if metrics.get("roc_curve"):
                    roc_data[layer_name] = metrics["roc_curve"]
        
        with open(output_dir / "roc_curves.json", 'w') as f:
            json.dump(roc_data, f, indent=2)
        
        print(f"   ✅ Saved ROC curves")
    
    # 3. Precision-Recall Curves
    if "metrics" in results and results["metrics"].get("pr_curve"):
        pr_data = {
            "overall": results["metrics"]["pr_curve"]
        }
        
        # Per-layer PR curves
        if "per_layer_metrics" in results:
            for layer_name, metrics in results["per_layer_metrics"].items():
                if metrics.get("pr_curve"):
                    pr_data[layer_name] = metrics["pr_curve"]
        
        with open(output_dir / "pr_curves.json", 'w') as f:
            json.dump(pr_data, f, indent=2)
        
        print(f"   ✅ Saved PR curves")
    
    # 4. Flow Diagram
    if "flow_diagram" in results:
        with open(output_dir / "flow_diagram.json", 'w') as f:
            json.dump(results["flow_diagram"], f, indent=2)
        
        print(f"   ✅ Saved flow diagram")
    
    # 5. Layer Statistics
    if "layer_stats" in results:
        with open(output_dir / "layer_stats.json", 'w') as f:
            json.dump(results["layer_stats"], f, indent=2)
        
        print(f"   ✅ Saved layer statistics")
    
    # 6. Flow Analysis
    if "flow_analysis" in results:
        with open(output_dir / "flow_analysis.json", 'w') as f:
            json.dump(results["flow_analysis"], f, indent=2)
        
        print(f"   ✅ Saved flow analysis")
    
    # 7. Summary metrics (for quick reference)
    if "metrics" in results:
        summary = {
            "test_mode": results.get("test_mode"),
            "strategy": results.get("strategy"),
            "num_samples": results.get("num_samples"),
            "accuracy": results["metrics"]["accuracy"],
            "precision": results["metrics"]["precision"],
            "recall": results["metrics"]["recall"],
            "f1_score": results["metrics"]["f1_score"],
            "fpr": results["metrics"]["fpr"],
            "fnr": results["metrics"]["fnr"]
        }
        
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   ✅ Saved summary")
    
    print(f"\n💾 All visualization data saved to: {output_dir}")


def load_visualization_data(viz_dir: Path) -> Dict:
    """
    Load visualization data from directory
    
    Returns:
        Dictionary with all visualization data
    """
    viz_dir = Path(viz_dir)
    data = {}
    
    files = {
        "confusion_matrices": "confusion_matrices.json",
        "roc_curves": "roc_curves.json",
        "pr_curves": "pr_curves.json",
        "flow_diagram": "flow_diagram.json",
        "layer_stats": "layer_stats.json",
        "flow_analysis": "flow_analysis.json",
        "summary": "summary.json"
    }
    
    for key, filename in files.items():
        filepath = viz_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data[key] = json.load(f)
    
    return data
