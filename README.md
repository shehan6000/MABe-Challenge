# MABe Challenge - Social Action Recognition in Mice

## 🎯 Overview

### Challenge Description

The MABe (Multi-Agent Behavior) Challenge focuses on automating the recognition of social behaviors in mice using pose estimation data. The goal is to identify 30+ different behaviors from tracking data of mice in various laboratory settings.



### Solution Approach

This solution provides two implementations:

1. **Baseline Model**: LightGBM with engineered features (faster, good baseline)
2. **Advanced Model**: Temporal Convolutional Network (TCN) for better sequence modeling

---

## 🏗️ Architecture

### High-Level Pipeline

```
Raw Pose Data → Feature Extraction → Temporal Modeling → Behavior Detection → Post-Processing → Submission
```

### Component Breakdown

```
MABe Pipeline
│
├── Data Loading Module
│   ├── load_metadata()
│   ├── load_tracking_data()
│   └── load_annotations()
│
├── Feature Engineering Module
│   ├── FeatureExtractor
│   │   ├── extract_spatial_features()
│   │   ├── compute_pairwise_features()
│   │   ├── add_temporal_features()
│   │   └── create_windows()
│   │
│   └── AdvancedFeatureExtractor
│       ├── extract_mouse_features()
│       ├── extract_interaction_features()
│       └── process_video()
│
├── Model Module
│   ├── BehaviorClassifier (LightGBM)
│   │   ├── prepare_training_data()
│   │   ├── train()
│   │   └── predict()
│   │
│   └── TCNBehaviorClassifier (PyTorch)
│       ├── TemporalBlock
│       └── forward()
│
└── Prediction & Submission Module
    ├── create_submission()
    └── post_process_predictions()
```

---



## 📊 Data Structure

### Metadata (train.csv / test.csv)

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | int | Unique video identifier |
| `lab_id` | string | Lab pseudonym (e.g., "CRIM13", "MABe22") |
| `mouse1_strain` | string | Mouse strain (e.g., "C57BL/6J") |
| `mouse1_color` | string | Mouse color marking |
| `mouse1_sex` | string | M/F |
| `mouse1_id` | string | Unique mouse ID |
| `frames_per_second` | int | Video frame rate |
| `video_duration_sec` | float | Video length in seconds |
| `pix_per_cm` | float | Pixel to cm conversion |
| `body_parts_tracked` | string | List of tracked body parts |
| `behaviors_labeled` | string | Behaviors annotated in this video |
| `tracking_method` | string | Pose estimation model used |

### Tracking Data (train_tracking/*.parquet)

| Column | Type | Description |
|--------|------|-------------|
| `video_frame` | int | Frame number (0-indexed) |
| `mouse_id` | string | Mouse identifier (e.g., "mouse1", "mouse2") |
| `bodypart` | string | Body part name (e.g., "nose", "left_ear", "tail_base") |
| `x` | float | X coordinate in pixels |
| `y` | float | Y coordinate in pixels |

**Example:**
```
video_frame, mouse_id, bodypart, x, y
0, mouse1, nose, 245.3, 156.7
0, mouse1, left_ear, 238.2, 149.1
0, mouse1, right_ear, 252.1, 149.5
0, mouse2, nose, 410.8, 320.2
...
```

### Annotation Data (train_annotation/*.parquet)

| Column | Type | Description |
|--------|------|-------------|
| `agent_id` | string | Mouse performing the behavior |
| `target_id` | string | Mouse receiving the behavior |
| `action` | string | Behavior name (e.g., "sniff", "chase", "groom") |
| `start_frame` | int | First frame of behavior |
| `stop_frame` | int | Last frame of behavior |

**Example:**
```
agent_id, target_id, action, start_frame, stop_frame
mouse1, mouse2, sniff, 100, 150
mouse2, mouse1, approach, 200, 250
mouse1, mouse1, groom, 300, 400
```

### Submission Format

```csv
row_id,video_id,agent_id,target_id,action,start_frame,stop_frame
0,101686631,mouse1,mouse2,sniff,0,10
1,101686631,mouse2,mouse1,approach,15,25
2,101686631,mouse1,mouse1,groom,30,50
```

---

## 🔨 Pipeline Components

### 1. Data Loading Module

#### `load_metadata()`
```python
def load_metadata():
    """
    Load training and test metadata files.
    
    Returns:
        tuple: (train_meta, test_meta) DataFrames
    
    Example:
        train_meta, test_meta = load_metadata()
        print(f"Training videos: {len(train_meta)}")
    """
```

#### `load_tracking_data(video_id, is_train=True)`
```python
def load_tracking_data(video_id, is_train=True):
    """
    Load pose tracking data for a specific video.
    
    Args:
        video_id (int): Video identifier
        is_train (bool): Whether this is training data
    
    Returns:
        DataFrame: Tracking data with columns [video_frame, mouse_id, bodypart, x, y]
        None: If file doesn't exist
    
    Example:
        tracking = load_tracking_data(101686631, is_train=True)
        print(tracking.head())
    """
```

#### `load_annotations(video_id)`
```python
def load_annotations(video_id):
    """
    Load behavior annotations for a training video.
    
    Args:
        video_id (int): Video identifier
    
    Returns:
        DataFrame: Annotations with columns [agent_id, target_id, action, start_frame, stop_frame]
        None: If file doesn't exist
    
    Example:
        annotations = load_annotations(101686631)
        print(f"Behaviors: {annotations['action'].unique()}")
    """
```

---

## 🎨 Feature Engineering

### Spatial Features

#### 1. **Basic Coordinates**
- Raw X, Y positions of each body part
- Centroid (center of mass) of each mouse

#### 2. **Distance Features**
```python
# Euclidean distance between mice
distance = sqrt((x1 - x2)² + (y1 - y2)²)

# Distance between specific body parts
nose_distance = sqrt((nose1_x - nose2_x)² + (nose1_y - nose2_y)²)
```

#### 3. **Angle Features**
```python
# Relative angle from mouse1 to mouse2
angle = arctan2(y2 - y1, x2 - x1)

# Body orientation (head to tail angle)
body_angle = arctan2(head_y - tail_y, head_x - tail_x)

# Angle difference (facing alignment)
angle_diff = abs(body_angle1 - body_angle2)
```

### Temporal Features

#### 1. **Velocity**
```python
velocity_x = x[t] - x[t-1]
velocity_y = y[t] - y[t-1]
speed = sqrt(velocity_x² + velocity_y²)
```

#### 2. **Acceleration**
```python
acceleration = speed[t] - speed[t-1]
```

#### 3. **Direction Change**
```python
direction = arctan2(velocity_y, velocity_x)
direction_change = direction[t] - direction[t-1]
```

### Interaction Features

#### 1. **Approach/Retreat**
```python
distance_change = distance[t] - distance[t-1]
is_approaching = (distance_change < 0)
```

#### 2. **Relative Velocity**
```python
relative_velocity = sqrt((v1_x - v2_x)² + (v1_y - v2_y)²)
```

#### 3. **Speed Ratio**
```python
speed_ratio = speed_agent / (speed_target + epsilon)
```

### Window-Based Features

#### Rolling Statistics (5, 10, 20 frame windows)
```python
# Mean distance over window
distance_mean_5 = distance.rolling(5, center=True).mean()

# Standard deviation (variability)
distance_std_10 = distance.rolling(10, center=True).std()

# Maximum/minimum values
distance_max_20 = distance.rolling(20, center=True).max()
```

### Feature Extraction Example

```python
# Extract features for a video
feature_extractor = FeatureExtractor()
tracking = load_tracking_data(video_id)

# Step 1: Compute pairwise features
pairwise = feature_extractor.compute_pairwise_features(tracking)

# Step 2: Add temporal features
pairwise = feature_extractor.add_temporal_features(pairwise)

# Step 3: Create windows for classification
windows = feature_extractor.create_windows(pairwise, window_size=30)
```

---

## 🤖 Model Architecture

### Baseline: LightGBM Classifier

#### Architecture
```
Input Features (n_features)
    ↓
LightGBM Gradient Boosting
    ├── Tree 1
    ├── Tree 2
    ├── ...
    └── Tree n
    ↓
Softmax Layer
    ↓
Behavior Predictions (n_classes)
```

#### Hyperparameters
```python
params = {
    'objective': 'multiclass',
    'num_class': n_behaviors,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}
```

#### Advantages
- Fast training and inference
- Handles missing values well
- Good feature importance analysis
- No GPU required

### Advanced: Temporal Convolutional Network (TCN)

#### Architecture
```
Input Sequence (batch, seq_len, features)
    ↓
Temporal Block 1 (dilation=1)
    ├── Conv1D + BatchNorm + ReLU
    ├── Conv1D + BatchNorm + ReLU
    └── Residual Connection
    ↓
Temporal Block 2 (dilation=2)
    ↓
Temporal Block 3 (dilation=4)
    ↓
Temporal Block 4 (dilation=8)
    ↓
Fully Connected Layer
    ↓
Softmax
    ↓
Behavior Predictions (batch, seq_len, n_classes)
```

#### Key Components

**Temporal Block:**
```python
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        # Dilated convolutions for exponentially growing receptive field
        # Residual connections for gradient flow
        # Batch normalization for stability
```

**Receptive Field:**
- Layer 1 (dilation=1): sees 3 frames
- Layer 2 (dilation=2): sees 7 frames
- Layer 3 (dilation=4): sees 15 frames
- Layer 4 (dilation=8): sees 31 frames

#### Advantages
- Captures long-range temporal dependencies
- Parallel processing (unlike RNNs)
- Exponentially growing receptive field
- Better for continuous behavior segmentation

---

## 🎓 Training Process

### Baseline Model Training

```python
# 1. Prepare data
X, y, video_ids = classifier.prepare_training_data(train_meta)

# 2. Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 3. Split by video (avoid data leakage)
unique_videos = np.unique(video_ids)
train_videos = unique_videos[:int(0.8 * len(unique_videos))]

# 4. Create datasets
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

# 5. Train
model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50)]
)
```

### Training Workflow

```
1. Load all training videos
    ↓
2. Extract features for each video
    ↓
3. Match features with annotations
    ↓
4. Create training samples
    ↓
5. Split by video (80/20)
    ↓
6. Train model
    ↓
7. Validate on held-out videos
    ↓
8. Save best model
```

### Cross-Validation Strategy

```python
# Group K-Fold by video
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=video_ids)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train model for this fold
    model = train_fold(X_train, y_train, X_val, y_val)
    
    # Evaluate
    score = evaluate(model, X_val, y_val)
    print(f"Fold {fold}: {score}")
```

---

## 🔮 Inference & Submission

### Prediction Pipeline

```
Test Video
    ↓
Load Tracking Data
    ↓
Extract Features
    ↓
Sliding Window Prediction
    ↓
Post-Processing
    ├── Merge consecutive predictions
    ├── Apply confidence threshold
    ├── Enforce minimum duration
    └── Smooth predictions
    ↓
Format Submission
    ↓
submission.csv
```

### Sliding Window Prediction

```python
# For each agent-target pair
for agent, target in pairs:
    # Get time-series features
    features = extract_features(agent, target)
    
    # Sliding window
    for start in range(0, len(features), stride):
        end = start + window_size
        window = features[start:end]
        
        # Predict
        behavior, confidence = model.predict(window)
        
        if confidence > threshold:
            predictions.append({
                'frame': center_frame,
                'behavior': behavior,
                'confidence': confidence
            })
```

### Post-Processing

#### 1. **Merge Consecutive Predictions**
```python
def merge_predictions(predictions):
    """
    Merge consecutive predictions of the same behavior.
    
    Example:
        Input:  [sniff(0-10), sniff(11-20), sniff(21-30)]
        Output: [sniff(0-30)]
    """
    merged = []
    current = None
    
    for pred in sorted(predictions, key=lambda x: x['start_frame']):
        if current is None:
            current = pred
        elif (pred['action'] == current['action'] and
              pred['start_frame'] - current['stop_frame'] <= 5):
            # Extend current behavior
            current['stop_frame'] = pred['stop_frame']
        else:
            merged.append(current)
            current = pred
    
    if current:
        merged.append(current)
    
    return merged
```

#### 2. **Confidence Filtering**
```python
# Only keep high-confidence predictions
predictions = [p for p in predictions if p['confidence'] > 0.6]
```

#### 3. **Minimum Duration**
```python
# Remove very short behaviors (likely noise)
predictions = [p for p in predictions 
               if p['stop_frame'] - p['start_frame'] >= 5]
```

#### 4. **Smoothing**
```python
# Apply Gaussian smoothing to prediction probabilities
smoothed_probs = gaussian_filter1d(probabilities, sigma=2)
```

---

## ⚙️ Configuration

### Config Class

```python
class Config:
    # Data paths
    DATA_PATH = Path('/kaggle/input/...')
    TRAIN_TRACKING_PATH = DATA_PATH / 'train_tracking'
    TEST_TRACKING_PATH = DATA_PATH / 'test_tracking'
    TRAIN_ANNOTATION_PATH = DATA_PATH / 'train_annotation'
    
    # Feature engineering
    WINDOW_SIZE = 30          # frames in sliding window
    STRIDE = 5                # overlap between windows
    
    # Post-processing
    MIN_BEHAVIOR_DURATION = 3 # minimum frames
    CONFIDENCE_THRESHOLD = 0.5
    
    # Model
    HIDDEN_DIM = 128
    NUM_TCN_LAYERS = 4
    KERNEL_SIZE = 3
    DROPOUT = 0.3
    
    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    
    # System
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_STATE = 42
```

### Customization

```python
# Modify for your experiments
Config.WINDOW_SIZE = 60  # Longer context
Config.STRIDE = 10       # Less overlap
Config.CONFIDENCE_THRESHOLD = 0.7  # More conservative
```

---

## 💻 Usage Examples

### Basic Usage

```python
# Run the complete pipeline
python main.py
```

### Step-by-Step Usage

```python
# 1. Load data
train_meta, test_meta = load_metadata()

# 2. Initialize components
classifier = BehaviorClassifier()
feature_extractor = FeatureExtractor()

# 3. Train
X, y, video_ids = classifier.prepare_training_data(train_meta)
classifier.train(X, y, video_ids)

# 4. Predict
submission = create_submission(test_meta, classifier, feature_extractor)

# 5. Save
submission.to_csv('submission.csv', index=False)
```

### Processing a Single Video

```python
# Load video data
video_id = 101686631
tracking = load_tracking_data(video_id)
annotations = load_annotations(video_id)

# Extract features
feature_extractor = FeatureExtractor()
features = feature_extractor.process_video(tracking)

# Inspect
print(f"Frames: {features['video_frame'].nunique()}")
print(f"Mouse pairs: {len(features[['agent_id', 'target_id']].drop_duplicates())}")
print(f"Features: {features.columns.tolist()}")
```

### Analyzing Predictions

```python
# Load submission
submission = pd.read_csv('submission.csv')

# Statistics
print(f"Total predictions: {len(submission)}")
print(f"Unique videos: {submission['video_id'].nunique()}")
print(f"Behaviors detected: {submission['action'].nunique()}")

# Behavior distribution
print(submission['action'].value_counts())

# Average behavior duration
submission['duration'] = submission['stop_frame'] - submission['start_frame']
print(f"Mean duration: {submission['duration'].mean():.1f} frames")
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. **Memory Error**
```python
# Problem: Running out of memory
# Solution: Process videos in batches

batch_size = 10
for i in range(0, len(train_meta), batch_size):
    batch = train_meta.iloc[i:i+batch_size]
    process_batch(batch)
```

#### 2. **Missing Body Parts**
```python
# Problem: Different labs track different body parts
# Solution: Use only common body parts or impute missing

def get_common_bodyparts(tracking):
    """Find body parts present for all mice"""
    bodyparts_by_mouse = tracking.groupby('mouse_id')['bodypart'].unique()
    common = set(bodyparts_by_mouse.iloc[0])
    for parts in bodyparts_by_mouse:
        common = common.intersection(set(parts))
    return list(common)
```

#### 3. **Slow Inference**
```python
# Problem: Prediction is too slow
# Solutions:
# 1. Increase stride
Config.STRIDE = 10  # Less overlap

# 2. Use vectorized operations
features = np.array([...])  # Batch processing

# 3. Reduce window size
Config.WINDOW_SIZE = 20  # Shorter sequences
```

#### 4. **Low F-Score**
```python
# Problem: Poor performance
# Solutions:
# 1. Check class balance
print(y.value_counts())

# 2. Add more features
# - Higher-order derivatives
# - Longer temporal windows
# - Cross-body-part distances

# 3. Tune confidence threshold
for threshold in [0.3, 0.5, 0.7]:
    Config.CONFIDENCE_THRESHOLD = threshold
    score = evaluate(model)
    print(f"Threshold {threshold}: {score}")
```

---

## 🚀 Advanced Topics

### 1. Ensemble Methods

```python
# Combine multiple models
models = [model_lgb, model_tcn, model_rf]

def ensemble_predict(models, X):
    predictions = []
    for model in models:
        pred = model.predict(X)
        predictions.append(pred)
    
    # Voting
    final_pred = mode(predictions, axis=0)
    return final_pred
```

### 2. Data Augmentation

```python
def augment_tracking(tracking):
    """Augment pose data"""
    # Horizontal flip
    tracking_flip = tracking.copy()
    tracking_flip['x'] = max(tracking['x']) - tracking['x']
    
    # Rotation
    angle = np.random.uniform(-15, 15)
    tracking_rot = rotate_coordinates(tracking, angle)
    
    # Scaling
    scale = np.random.uniform(0.9, 1.1)
    tracking_scale = tracking.copy()
    tracking_scale[['x', 'y']] *= scale
    
    return [tracking, tracking_flip, tracking_rot, tracking_scale]
```

### 3. Multi-Task Learning

```python
# Predict behavior AND lab simultaneously
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = TCN(...)
        self.behavior_head = nn.Linear(hidden_dim, n_behaviors)
        self.lab_head = nn.Linear(hidden_dim, n_labs)
    
    def forward(self, x):
        features = self.shared(x)
        behavior = self.behavior_head(features)
        lab = self.lab_head(features)
        return behavior, lab
```

### 4. Attention Mechanisms

```python
class AttentionLayer(nn.Module):
    """Attention over time steps"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        scores = self.attention(x)  # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context
```

### 5. Domain Adaptation

```python
# Learn lab-invariant features
class DomainAdversarialModel(nn.Module):
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.behavior_classifier = BehaviorHead()
        self.domain_classifier = DomainHead()
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # Predict behavior
        behavior = self.behavior_classifier(features)
        
        # Predict domain (with gradient reversal)
        domain = self.domain_classifier(GradientReversal()(features))
        
        return behavior, domain
```
