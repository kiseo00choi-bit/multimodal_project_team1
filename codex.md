# codex.md

## Project Overview

This project implements an abnormal behavior classification system for unmanned store CCTV videos.

The goal is to compare three baseline models and one proposed multimodal fusion model:

1. CNN + Average Pooling baseline
2. CNN + LSTM baseline
3. 1D-CNN + LSTM keypoint baseline
4. RGB + Keypoint Fusion proposed model
5. RGB + Keypoint Cross-Attention Fusion extension

The project uses indoor store CCTV video data and XML annotation data. The XML labels contain action classes, action start/end frames, bounding boxes, and human keypoint coordinates. The final objective is to evaluate whether combining RGB video features and human pose keypoint features improves abnormal behavior classification performance compared with single-modal baselines.

## Target Classes

The classification target consists of eight abnormal behavior classes:

- fall: person falling down
- broken: object/property damage
- fire: arson or fire-related behavior
- smoke: smoking behavior
- abandon: abandoned object behavior
- theft: stealing behavior
- fight: physical violence
- weak pedestrian: mobility-impaired pedestrian behavior

## Expected Data Structure

The dataset is expected to contain:

- MP4 video files
- XML label files

Each XML file may contain:

- action class label
- action start frame
- action end frame
- person bounding box
- frame-level keypoint coordinates

A single training sample should represent one action segment rather than a full raw video.

## Overall Pipeline

Follow this project flow:

```text
Raw MP4 video + XML label
??XML parsing
??action segment extraction
??frame sampling
??RGB frame preprocessing
??keypoint extraction and normalization
??dataset split
??baseline model training
??fusion model training
??evaluation and comparison
```

## Data Preprocessing Flow

### 1. XML Parsing

Parse each XML file and extract:

- class label
- start frame
- end frame
- bounding box coordinates if available
- keypoint coordinates for each frame if available

The parser should return structured metadata per action segment.

Recommended output format per sample:

```python
{
    "video_path": str,
    "xml_path": str,
    "label": str,
    "start_frame": int,
    "end_frame": int,
    "bbox_sequence": Optional[list],
    "keypoint_sequence": Optional[list]
}
```

### 2. Frame Sampling

For each action segment, sample a fixed number of frames from `start_frame` to `end_frame`.

Recommended default:

- `num_frames = 16` or `32`
- use uniform sampling across the action segment
- if the segment is shorter than the target frame count, repeat or pad frames

The model should receive fixed-length sequences.

### 3. RGB Frame Preprocessing

Each sampled RGB frame should be converted into a tensor.

Recommended preprocessing:

- resize frame to `224 x 224`
- normalize with ImageNet mean/std if using pretrained CNN backbones
- output shape: `[T, C, H, W]`

Where:

- `T`: number of sampled frames
- `C`: RGB channels, usually 3
- `H`, `W`: image height and width

### 4. Keypoint Preprocessing

Each frame contains 17 human joints.

Use only x, y coordinates unless confidence scores are clearly available and consistently formatted.

```text
17 joints 횞 (x, y) = 34-dimensional vector per frame
```

For each action segment:

```text
T frames 횞 34 dimensions = keypoint sequence
```

Recommended normalization:

- normalize x by image width
- normalize y by image height
- missing keypoints should be handled consistently by zero padding or interpolation
- output shape: `[T, 34]`

## Dataset Split Rule

Split data into:

- train
- validation
- test

Important rule:

Do not split frames from the same action segment into different sets.
A full action segment should belong to only one split.

Preferred split level:

1. video-level split if multiple clips come from the same raw video
2. otherwise action-segment-level split

## Model Experiments

## Baseline 1: CNN + Average Pooling

### Purpose

This is the simplest RGB-only baseline.
It checks how well abnormal behaviors can be classified using frame-level visual features only.

### Input

```text
RGB frame sequence: [T, C, H, W]
```

### Architecture

```text
RGB frames
??CNN feature extractor
??frame feature sequence
??average pooling over time
??fully connected classifier
??class prediction
```

### Implementation Notes

- Use a CNN backbone such as ResNet18 or MobileNetV3.
- Extract a feature vector for each frame.
- Average the frame features along the time dimension.
- Feed the pooled feature into a classifier.

### Limitation

This model does not strongly model temporal order.
It can capture visual appearance but not detailed motion flow.

## Baseline 2: CNN + LSTM

### Purpose

This baseline checks whether modeling temporal order improves performance compared with simple average pooling.

### Input

```text
RGB frame sequence: [T, C, H, W]
```

### Architecture

```text
RGB frames
??CNN feature extractor
??frame feature sequence
??LSTM or GRU
??video feature
??fully connected classifier
??class prediction
```

### Implementation Notes

- Use the same CNN backbone as Baseline 1 when possible.
- Keep CNN feature dimension consistent across RGB baselines.
- Feed frame features into an LSTM in chronological order.
- Use the final hidden state or pooled hidden states for classification.

### Expected Strength

This model is expected to work better for motion-dependent classes such as:

- fall
- fight
- theft

## Baseline 3: 1D-CNN + LSTM Keypoint Model

### Purpose

This is the pose-only baseline.
It checks how well abnormal behaviors can be classified using only human body joint movement.

### Input

```text
Keypoint sequence: [T, 34]
```

### Architecture

```text
keypoint sequence
??1D-CNN over temporal axis
??LSTM or GRU
??pose feature
??fully connected classifier
??class prediction
```

### Implementation Notes

- Treat the keypoint sequence as temporal data.
- Use 1D-CNN to capture local motion patterns.
- Use LSTM to capture longer temporal posture changes.
- Normalize keypoints before model input.

### Expected Strength

This model may perform well on posture-sensitive classes such as:

- fall
- fight
- weak pedestrian

### Expected Weakness

This model may struggle with object-dependent classes such as:

- smoke
- theft
- fire
- broken

because object appearance and scene context may be important.

## Proposed Model: RGB + Keypoint Fusion

### Purpose

This is the final proposed multimodal model.
It combines visual scene information from RGB frames and pose/motion information from keypoints.

### Inputs

```text
RGB frame sequence: [T, C, H, W]
Keypoint sequence: [T, 34]
```

### Architecture

```text
[RGB Branch]
RGB frames
??CNN feature extractor
??frame feature sequence
??LSTM
??video feature

[Pose Branch]
keypoint sequence
??1D-CNN
??LSTM
??pose feature

[Fusion]
concat(video feature, pose feature)
??fully connected layers
??class prediction
```

### Fusion Rule

Use feature-level fusion by default.

Recommended default:

```python
fused_feature = torch.cat([video_feature, pose_feature], dim=-1)
```

Then pass `fused_feature` to a classification head.

### Expected Benefit

RGB frames provide:

- object information
- scene context
- appearance cues

Keypoints provide:

- posture information
- motion pattern
- body movement sequence

The fusion model should be compared with all three baselines to verify whether multimodal input improves classification performance.

## Training Flow

Use the same training pipeline for all models when possible.

```text
load dataset
??create dataloader
??initialize model
??train for each epoch
??validate after each epoch
??save best checkpoint by validation Macro F1-score
??evaluate best checkpoint on test set
```

Recommended training settings:

- loss: CrossEntropyLoss
- optimizer: Adam or AdamW
- scheduler: optional CosineAnnealingLR or ReduceLROnPlateau
- batch size: depends on GPU memory
- epoch: start with 20 to 30
- metric for best model: validation Macro F1-score

## Evaluation Metrics

Evaluate every experiment using the same metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1-score
- Confusion Matrix

Important:

Accuracy alone is not enough because class imbalance may exist.
Macro F1-score and class-level confusion matrix must be included in the final comparison.

## Required Result Table

The final result should include a table like this:

| Experiment | Input | Model | Accuracy | Macro F1 |
|---|---|---|---:|---:|
| Baseline 1 | RGB frames | CNN + Average Pooling | TBD | TBD |
| Baseline 2 | RGB frames | CNN + LSTM | TBD | TBD |
| Baseline 3 | Keypoints | 1D-CNN + LSTM | TBD | TBD |
| Proposed | RGB + Keypoints | Fusion Model | TBD | TBD |

## Error Analysis

After evaluation, analyze:

- which classes are frequently confused
- whether RGB-based models perform better on object-related actions
- whether keypoint-based models perform better on posture-related actions
- whether fusion improves weak classes
- whether errors come from poor frame sampling, missing keypoints, or ambiguous labels

## Current Project Structure

Use this structure for the current workspace. The AI Hub zip files were extracted
into `data/extracted`, and the original zip folder `data/raw` was deleted to save
SSD space.

```text
team_project/
|-- codex.md
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- configs/
|   |-- baseline_cnn_avg.yaml
|   |-- baseline_cnn_lstm.yaml
|   |-- baseline_keypoint.yaml
|   `-- fusion.yaml
|-- data/
|   |-- extracted/
|   |   |-- Training/
|   |   |   |-- videos/
|   |   |   `-- labels/
|   |   `-- Validation/
|   |       |-- videos/
|   |       `-- labels/
|   |-- processed/
|   `-- splits/
|       `-- manifest.csv
|-- docs/
|   `-- work_log_ko.md
|-- scripts/
|   |-- extract_aihub_dataset.py
|   |-- build_manifest.py
|   `-- inspect_video_fps.py
|-- src/
|   |-- data/
|   |   |-- xml_parser.py
|   |   |-- frame_sampler.py
|   |   |-- preprocessing.py
|   |   `-- dataset.py
|   |-- models/
|   |   |-- cnn_avg.py
|   |   |-- cnn_lstm.py
|   |   |-- keypoint_lstm.py
|   |   `-- fusion.py
|   |-- train.py
|   |-- evaluate.py
|   `-- utils.py
|-- outputs/
|   |-- checkpoints/
|   |-- logs/
|   |-- metrics/
|   `-- figures/
|-- notebooks/
`-- .venv5070/
```

## Current Dataset Notes

The extracted dataset is the active training source.

```text
dataset_root: data/extracted
manifest_path: data/splits/manifest.csv
```

Verified data summary:

```text
MP4 files: 5,841
XML files: 5,841
Missing XML pairs: 0
Video FPS: 3.0 for all videos
Typical duration: about 60 seconds
Typical frame count: 180 or 181 frames
Known frame mismatch: 1 validation smoking sample has 180 video frames but XML size 181
```

Processed RGB frame summary:

```text
Frame root: data/processed/frames_224
Frame image size: 224x224 RGB JPEG
JPEG frames: 1,052,905
Video-level manifest: data/processed/frames_224_manifest.csv
Frame-level index: data/processed/frame_index_224.csv
Label map: data/processed/label_map.json
Frame directory mismatches: 0
Processed frame data size: about 19.65GB
```

Training and automatic test-evaluation entry point:

```powershell
.\.venv5070\Scripts\python.exe scripts\run_all_experiments.py
```

Smoke test command:

```powershell
.\.venv5070\Scripts\python.exe scripts\run_all_experiments.py --smoke
```

The all-experiment runner trains in this order:

```text
baseline_cnn_avg -> baseline_cnn_lstm -> baseline_keypoint -> fusion
```

Current split rule:

```text
AI Hub Training   -> train / val with fixed stratified split
AI Hub Validation -> test only
Split manifest: data/processed/frames_224_trainvaltest.csv
```

After full training, run test evaluation:

```powershell
.\.venv5070\Scripts\python.exe scripts\evaluate_all_best.py
```

Current training stability note:

- The active split names are `train`, `val`, and `test`.
- The training DataLoader must shuffle both `train` and legacy `Training` split names.
- A previous non-converging CNN baseline run was traced to the shuffle condition still checking only `Training`.
- RGB models use frozen ImageNet pretrained ResNet18 features by default, followed by LayerNorm before the temporal or classification head.

Learning curves are saved automatically after each training run:

```text
outputs/figures/*_learning_curve.png
```

To regenerate curves from `outputs/metrics/*_history.csv`:

```powershell
.\.venv5070\Scripts\python.exe scripts\plot_learning_curves.py
```

Class mapping:

```text
07: fall / 전도
08: broken / 파손
09: fire / 방화
10: smoke / 흡연
11: abandon / 유기
12: theft / 절도
13: fight / 폭행
14: weak pedestrian / 교통약자
```

When implementing the dataset loader, use `manifest.csv` as the source of truth
for video/XML pairing. Clamp XML frame indices to the actual video frame count to
handle the one known off-by-one sample.

## Coding Rules for Codex

Follow these rules when editing or generating code:

- Use Python and PyTorch by default.
- Keep model, dataset, training, and evaluation code separated.
- Do not hard-code absolute dataset paths inside source files.
- Use config files or command-line arguments for paths and hyperparameters.
- Save metrics as CSV or JSON.
- Save confusion matrix figures under `outputs/figures/`.
- Save checkpoints under `outputs/checkpoints/`.
- Use clear English variable and function names.
- Write comments only where the logic is not obvious.
- Do not commit raw video data or large processed tensors.

## Implementation Priority

Implement in this order:

1. XML parser
2. frame sampler
3. keypoint extractor and normalizer
4. dataset class returning RGB frames, keypoints, and labels
5. Baseline 1: CNN + Average Pooling
6. Baseline 2: CNN + LSTM
7. Baseline 3: 1D-CNN + LSTM keypoint model
8. Proposed Fusion model
9. common train/evaluate scripts
10. metrics and confusion matrix visualization

## Dataset Class Requirement

The dataset should support all experiments with one unified class if possible.

Recommended return format:

```python
sample = {
    "frames": frames_tensor,        # [T, C, H, W]
    "keypoints": keypoints_tensor,  # [T, 34]
    "label": label_tensor,
    "metadata": metadata_dict
}
```

For RGB-only models, ignore `keypoints`.
For keypoint-only models, ignore `frames`.
For fusion models, use both.

## Reproducibility Rules

Set random seeds for:

- Python random
- NumPy
- PyTorch
- CUDA if available

Save the following for every run:

- config file
- model name
- random seed
- train/validation/test split path
- best validation score
- test metrics
- confusion matrix

## Presentation/Report Requirements

Final report and presentation should explain:

1. project motivation
2. dataset structure
3. preprocessing flow
4. baseline models
5. proposed fusion model
6. experiment settings
7. result comparison
8. confusion matrix analysis
9. limitations
10. future improvements

Use Korean polite report style for final written deliverables.
Prefer `?듬땲???낅땲?? tone.

## Team Role Mapping

### ?댁???
Responsible for data structure analysis and preprocessing.

Main tasks:

- inspect MP4 and XML structure
- implement XML parser
- extract labels, frame ranges, bounding boxes, and keypoints
- implement frame sampling
- build train/validation/test split

### ?뺥삙由?
Responsible for RGB video baseline models.

Main tasks:

- implement CNN + Average Pooling baseline
- implement CNN + LSTM baseline
- compare RGB-only baseline performance
- organize RGB model results

### 理쒓린??
Responsible for keypoint baseline and fusion model.

Main tasks:

- normalize keypoint sequences
- implement 1D-CNN + LSTM keypoint model
- implement RGB + Keypoint Fusion model
- compare fusion model with baselines

### Common Tasks

All members should participate in:

- result table creation
- confusion matrix interpretation
- error case analysis
- final report writing
- presentation slide preparation

## Things Codex Should Avoid

- Do not rewrite the whole project structure unless necessary.
- Do not remove existing working code without explanation.
- Do not mix preprocessing, model definition, and training logic in one large file.
- Do not assume every XML file has perfect keypoints.
- Do not evaluate only with accuracy.
- Do not use test data during model selection.
- Do not include API keys, local passwords, or private paths in committed files.

## First Task Recommendation

If starting from an empty or incomplete repository, begin by creating:

1. `src/data/xml_parser.py`
2. `src/data/frame_sampler.py`
3. `src/data/dataset.py`
4. a small debug script that loads one video/XML pair and prints parsed label, sampled frame count, keypoint shape, and class label

The first working milestone is:

```text
One sample can be loaded as:
frames: [T, C, H, W]
keypoints: [T, 34]
label: class index
```

After this milestone, implement the models.
