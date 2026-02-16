# Fabric Defect Segmentation Report

## Dataset Characteristics

- **Class imbalance**: Micro/macro defects have limited examples
- **Task type**: Semantic segmentation (defects are elongated and better suited for segmentation than detection/instance segmentation)
- **Selected classes**: Broken defects (end, yarn, pick) due to similar distribution
- **Spatial constraints**: Classes exhibit directional features (vertical/horizontal), limiting use of rotation augmentations
- **Baseline model**: binary segmentation on all 3 classes

## Current Model Performance

- **IoU metric**: 0.96 on training/validation data
- **Overfitting**: Model memorizes training/validation sets
- **Generalization**: Poor performance on held-out test images

## Improvement Strategies for Multiclass Models

### Regularization
- Implement combined loss functions
- Use AdamW optimizer for better weight decay
- Apply k-fold cross-validation

### Data Enhancement
- Improve augmentation pipeline (moderate, non-aggressive transforms)
- Annotate additional data
- Generate synthetic training examples

## Alternative Approaches

### Advanced Architectures
- SAM3 (Segment Anything Model v3)
- Transformer-based models

### Defect Detection Paradigms

**Option 1: Multi-class defect model**
- Train on all defect classes simultaneously
- **Challenges**: Class imbalance (micro/macro defects), potential class-specific overfitting

**Option 2: Anomaly detection via fabric type learning**
- Train model on defect-free fabric types
- Invert masks and crop image edges
- Augment normal fabric to simulate defects

### Ensemble Strategy
- **Model 1**: Segment defect-free fabric types
- **Model 2**: Detect defects/anomalies
- Combine outputs for robust inference