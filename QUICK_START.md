# SICKLE++ Quick Start (5 Minutes)

Get up and running with SICKLE++ baseline in 5 minutes.

## Option 1: Demo with Synthetic Data (FASTEST)

```bash
# Clone/navigate to project
cd sickle-plus-plus

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install torch torchvision scikit-learn matplotlib seaborn numpy pandas tqdm

# Run demo
python scripts/demo.py
```

**What happens**:
- Creates 50 synthetic temporal sequences
- Trains UTAE model for 3 epochs
- Evaluates on test set
- Generates results and visualizations

**Expected time**: ~2-3 minutes on GPU, ~5-10 minutes on CPU

**Outputs**:
```
demo_data/                  # Synthetic satellite data
demo_checkpoints/           # Trained model
demo_results/               # Evaluation results and plots
```

---

## Option 2: Train on Your Data (10 MINUTES)

### Step 1: Prepare Data
```
your_data_dir/
├── satellite_001.npy  (shape: T×C×H×W)
├── satellite_002.npy
├── ...
└── labels.json
    {"satellite_001": 5, "satellite_002": 12}
```

### Step 2: Train
```bash
python scripts/train.py \
    --data_dir ./your_data_dir \
    --labels_path ./your_data_dir/labels.json \
    --model utae \
    --epochs 50 \
    --batch_size 8 \
    --output_dir ./my_checkpoints
```

### Step 3: Evaluate
Results automatically saved to `./my_checkpoints/test_results/`

---

## Option 3: Run Inference on Pre-trained Model

```bash
python scripts/infer.py \
    --model_path ./demo_checkpoints/best_model.pth \
    --data_path ./satellite_data.npy \
    --output_dir ./results
```

Output: `./results/inference_results.json`

---

## API Usage (Python)

```python
import numpy as np
from pipeline.models.inference import ModelInference, SICKLEInference

# Load pre-trained model
model = ModelInference(
    model_path='./demo_checkpoints/best_model.pth',
    model_type='utae'
)

# Load satellite data
data = np.load('satellite_data.npy')  # Shape: (T, C, H, W)

# Run inference
results = model.predict(data)
print(f"Predicted class: {results['class_predictions']}")
print(f"Confidence: {results['mean_confidence']:.3f}")

# Agricultural-specific analysis
sickle = SICKLEInference('./demo_checkpoints/best_model.pth')
field_results = sickle.predict_field(data)
print(f"Dominant crop: {field_results['dominant_crop']}")
```

---

## Training Script (Full Control)

```python
import torch
from pipeline.datasets import SICKLEDataset, create_dataloaders
from pipeline.training import SICKLETrainer, TrainingConfig
from models.utae import UTAE

# Load data
dataset = SICKLEDataset('./data', './labels.json')
train_loader, val_loader, test_loader = create_dataloaders(
    dataset, batch_size=8, num_workers=4
)

# Create model
model = UTAE(
    input_dim=21,
    encoder_widths=[64, 64, 64, 128],
    decoder_widths=[32, 32, 64, 128],
    out_conv=[32, 20],
    n_head=16,
    d_model=256
)

# Configure training
config = TrainingConfig(
    max_epochs=50,
    batch_size=8,
    learning_rate=1e-4,
    optimizer='adamw',
    patience=10
)

# Train
trainer = SICKLETrainer(model, config=config, output_dir='./checkpoints')
trainer.train(train_loader, val_loader)

# Best model saved to: ./checkpoints/best_model.pth
```

---

## Data Formats

### Input Data (Satellite)

**Single-time (Classification)**:
```python
shape: (C, H, W)
C = 21 channels
  - Sentinel-2: 12 bands
  - Sentinel-1: 2 bands
  - Landsat-8: 7 bands
H, W >= 32 (patch size)
dtype: float32
range: [0, 1] (normalized)
```

**Temporal (Time Series)**:
```python
shape: (T, C, H, W)
T = 61 frames (standard)
C = 21 channels
H, W >= 32
dtype: float32
```

### Labels

**JSON** (recommended):
```json
{"image_001": 5, "image_002": 12}
```

**CSV**:
```
filename,label
image_001.npy,5
image_002.npy,12
```

**NPY**:
```python
np.array([5, 12, ...])  # Shape: (n_samples,)
```

---

## Available Models

| Model | Best For | Speed | Memory |
|-------|----------|-------|--------|
| **UTAE** | Temporal classification | Fast | Low (2M params) |
| **UNet3D** | Dense prediction | Medium | Medium (15M) |
| **PaSTiS-UNet3D** | Long sequences | Slow | High (20M) |

Default: **UTAE** (recommended for baseline)

---

## Troubleshooting

### OutOfMemory
```bash
# Reduce batch size
--batch_size 4

# Use CPU
--device cpu
```

### Poor Performance
1. **Check data**: Verify cloud masking, look at distribution
2. **Increase epochs**: Try --epochs 100
3. **Tune learning rate**: Try 1e-5 or 1e-3

### CUDA Issues
```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU
python scripts/train.py ... --device cpu
```

### Missing Dependencies
```bash
pip install torch torchvision
pip install scikit-learn matplotlib seaborn
pip install earthengine-api geopandas
```

---

## Expected Results

**UTAE Baseline on PASTIS**:
- Accuracy: 75-82%
- F1 Score: 0.73-0.80
- Training time: 2-4 hours (GPU)

**Paper Target**: 70-85% ✅

---

## Common Tasks

### 1. Use Different Model
```bash
python scripts/train.py ... --model unet3d
# or: pastis, unet3d_multitask
```

### 2. Different Task
```bash
python scripts/train.py ... --task yield
# or: phenology, crop_classification
```

### 3. Save/Load Checkpoint
```python
# Save
torch.save(model.state_dict(), 'checkpoint.pth')

# Load
model.load_state_dict(torch.load('checkpoint.pth'))
```

### 4. Evaluate with Labels
```bash
python scripts/infer.py \
    --model_path best_model.pth \
    --data_path test_data.npy \
    --labels_path test_labels.npy  # Add this
```

---

## Next Steps

1. ✅ Run demo.py
2. 📊 Prepare your satellite data
3. 🚀 Train with train.py
4. 🔍 Analyze results in output_dir
5. 🎯 Improve model (advanced: temporal transformers, fusion)

---

## File Locations

| Component | Location |
|-----------|----------|
| Training script | `scripts/train.py` |
| Inference script | `scripts/infer.py` |
| Demo | `scripts/demo.py` |
| Models | `models/utae.py`, etc. |
| Datasets | `pipeline/datasets/` |
| Training utils | `pipeline/training.py` |
| Metrics | `pipeline/evaluation.py` |
| Guide | `BASELINE_SETUP.md` |
| Summary | `CODEBASE_SUMMARY.md` |

---

## Getting Help

1. **Run demo**: `python scripts/demo.py`
2. **Read guide**: `BASELINE_SETUP.md`
3. **Check code**: Inline docs in Python files
4. **Check logs**: `checkpoints/logs/training_logs.json`

---

**Status**: ✅ Ready to use
**Time commitment**: Demo 3min, Training 2-4h (GPU)
**Support**: Fully documented Python codebase

Good luck! 🚀
