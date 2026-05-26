# Abnormal Behavior Classification

실내 무인 매장 CCTV 영상과 XML 라벨링 데이터를 활용해 이상행동을 분류하는 딥러닝 프로젝트입니다. AI Hub 데이터의 RGB 영상과 사람 keypoint 라벨을 사용하여 RGB baseline, keypoint baseline, RGB+keypoint fusion 모델을 비교했습니다.

## Environment

RTX 5070 Ti 환경에 맞춰 `.venv5070` 가상환경을 사용했습니다.

```powershell
.\.venv5070\Scripts\Activate.ps1
```

확인된 주요 환경은 다음과 같습니다.

```text
Python: 3.11.9
PyTorch: 2.11.0+cu128
CUDA: 12.8
GPU: NVIDIA GeForce RTX 5070 Ti
```

## Dataset

사용 데이터는 AI Hub의 실내 편의점/매장 사람 이상행동 데이터입니다. 원본 zip은 약 253GB였고, 압축 해제 후 MP4 영상과 XML 라벨 파일만 유지했습니다. 원본 zip은 중복 저장을 피하기 위해 삭제했습니다.

데이터 설명 페이지:

```text
https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%ED%8E%B8%EC%9D%98%EC%A0%90&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71550
```

### 가공 전 데이터

AI Hub 원본 데이터는 `Training`과 `Validation`으로 나뉘어 있었고, 각 split 내부에 원천데이터 MP4와 라벨링데이터 XML zip이 포함되어 있었습니다.

```text
data/raw/
`-- 01-1.정식개방데이터/
    |-- Training/
    |   |-- 01.원천데이터/
    |   `-- 02.라벨링데이터/
    `-- Validation/
        |-- 01.원천데이터/
        `-- 02.라벨링데이터/
```

압축 해제 및 정리 후에는 다음 구조로 변환했습니다.

```text
data/extracted/
|-- Training/
|   |-- videos/
|   `-- labels/
`-- Validation/
    |-- videos/
    `-- labels/
```

데이터 검증 결과는 다음과 같습니다.

```text
MP4 files: 5,841
XML files: 5,841
Missing XML: 0
Video FPS: 3.0 for all videos
Typical duration: about 60 seconds
Typical frame count: 180 or 181 frames
Extracted data size: about 255.63GB
```

### 가공 후 데이터

CNN/LSTM 학습을 위해 모든 MP4 영상을 224x224 RGB JPEG 프레임으로 변환했습니다.

```text
data/processed/frames_224/
|-- Training/
|   |-- 07_fall/
|   |-- 08_broken/
|   |-- ...
`-- Validation/
    |-- 07_fall/
    |-- 08_broken/
    |-- ...
```

전처리 결과는 다음과 같습니다.

```text
Frame image size: 224x224 RGB
JPEG frames: 1,052,905
Videos: 5,841
Frame directory mismatches: 0
Processed frame data size: about 19.65GB
```

224x224로 변환한 이유는 ImageNet pretrained ResNet18의 표준 입력 크기와 호환성이 좋고, 원본 1920x1080 영상을 그대로 사용할 때 발생하는 GPU 메모리와 학습 시간 부담을 줄이기 위해서입니다.

학습용 매칭 파일은 다음과 같습니다.

```text
data/processed/frames_224_manifest.csv       # video-level metadata
data/processed/frames_224_trainvaltest.csv  # fixed train/val/test split
data/processed/frame_index_224.csv          # frame-level image index
data/processed/label_map.json               # class id/name mapping
```

## Classes

모델 학습에는 0부터 시작하는 `class_id`를 사용했습니다. 보고서에서는 사람이 읽기 쉽도록 1번부터 표시할 수 있습니다.

| class_id | 보고서 번호 | AI Hub code | English label | Korean label |
|---:|---:|---|---|---|
| 0 | 1 | 07 | fall | 전도 |
| 1 | 2 | 08 | broken | 파손 |
| 2 | 3 | 09 | fire | 방화 |
| 3 | 4 | 10 | smoke | 흡연 |
| 4 | 5 | 11 | abandon | 유기 |
| 5 | 6 | 12 | theft | 절도 |
| 6 | 7 | 13 | fight | 폭행 |
| 7 | 8 | 14 | weak_pedestrian | 교통약자 |

## Split

초기 실험에서는 AI Hub `Training`을 학습에 사용하고 AI Hub `Validation`을 validation으로 사용했지만, 이 방식은 최종 test set이 따로 없는 문제가 있었습니다. 최종 실험에서는 다음 구조로 수정했습니다.

```text
AI Hub Training   -> train / val stratified split
AI Hub Validation -> test only
```

최종 split 수는 다음과 같습니다.

```text
train: 4,154 samples
val:   1,037 samples
test:    650 samples
```

모델 선택은 validation Macro F1-score 기준으로 수행하고, test set은 선택된 best checkpoint의 최종 평가에만 사용했습니다.

## Models

비교한 모델은 4개입니다.

| Experiment | Input | Architecture | Pretrained |
|---|---|---|---|
| Baseline 1 | RGB frames | ResNet18 + Average Pooling + FC | ImageNet ResNet18 |
| Baseline 2 | RGB frames | ResNet18 + GRU + FC | ImageNet ResNet18 |
| Baseline 3 | Keypoints | 1D-CNN + GRU + FC | 없음 |
| Proposed | RGB + Keypoints | ResNet18+GRU branch, 1D-CNN+GRU branch, feature concat | ImageNet ResNet18 for RGB |
| Proposed v2 | RGB + Keypoints | Keypoint-query Cross-Attention over RGB + GRU + FC | ImageNet ResNet18 for RGB |

RGB 계열 모델은 `torchvision.models.resnet18`의 ImageNet pretrained weight를 사용했습니다. 현재 설정에서는 backbone을 feature extractor로 고정하고, 추출된 feature에 `LayerNorm`을 적용한 뒤 temporal module 또는 classifier로 전달합니다.

Keypoint 모델은 XML에서 추출한 17개 관절의 x, y 좌표를 사용합니다. 입력 shape은 `[T, 34]`이며, 좌표는 원본 영상 width/height 기준으로 0~1 범위로 정규화했습니다.

## Experiment Flow

전체 실험 과정은 다음 순서로 진행했습니다.

```text
1. AI Hub zip 이동 및 압축 해제
2. MP4/XML 파일 구조 정리
3. 영상 FPS, frame count, XML size 검증
4. MP4와 XML 1:1 매칭 manifest 생성
5. 전체 영상을 224x224 RGB JPEG 프레임으로 변환
6. XML에서 action frame range와 keypoint sequence 파싱
7. AI Hub Training을 train/val로 stratified split
8. 4개 모델을 동일한 train loop로 순차 학습
9. validation Macro F1-score 기준 best checkpoint 저장
10. best checkpoint를 AI Hub Validation 기반 test split에서 최종 평가
11. history CSV, summary JSON, confusion matrix, learning curve 저장
```

전체 실험 실행 명령은 다음과 같습니다.

```powershell
.\.venv5070\Scripts\python.exe scripts\run_all_experiments.py
```

작은 데이터로 pipeline만 확인하려면 다음 명령을 사용합니다.

```powershell
.\.venv5070\Scripts\python.exe scripts\run_all_experiments.py --smoke
```

결과 저장 위치는 다음과 같습니다.

```text
outputs/checkpoints/
outputs/metrics/
outputs/figures/
outputs/runs/
```

대용량 데이터와 학습 결과물은 git에는 포함하지 않습니다.

## Results

최종 결과는 `outputs/metrics/all_experiments_full.json` 기준입니다. 표의 validation 성능은 best checkpoint 선택에 사용된 점수이고, test 성능은 선택된 checkpoint를 AI Hub Validation 기반 test split에서 평가한 결과입니다.

| Experiment | Best Epoch | Valid Macro F1 | Test Accuracy | Test Macro F1 |
|---|---:|---:|---:|---:|
| CNN + Average Pooling | 19 | 0.8263 | 0.6323 | 0.6246 |
| CNN + GRU | 17 | 0.9611 | 0.7554 | 0.7547 |
| 1D-CNN + GRU Keypoint | 30 | 0.9864 | 0.9492 | 0.9497 |
| RGB + Keypoint Fusion | 21 | 0.9805 | 0.8523 | 0.8467 |
| RGB + Keypoint Cross-Attention Fusion | 27 | 0.9826 | 0.8646 | 0.8620 |

### Test Class-wise F1-score

| Experiment | 전도 | 파손 | 방화 | 흡연 | 유기 | 절도 | 폭행 | 교통약자 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN + Average Pooling | 0.821 | 0.300 | 0.783 | 0.697 | 0.345 | 0.323 | 0.729 | 1.000 |
| CNN + GRU | 0.937 | 0.490 | 0.988 | 0.859 | 0.656 | 0.390 | 0.719 | 1.000 |
| 1D-CNN + GRU Keypoint | 1.000 | 0.906 | 0.975 | 0.901 | 0.942 | 0.905 | 0.981 | 0.988 |
| RGB + Keypoint Fusion | 1.000 | 0.654 | 0.982 | 0.785 | 0.969 | 0.513 | 0.871 | 1.000 |
| RGB + Keypoint Cross-Attention Fusion | 0.981 | 0.684 | 1.000 | 0.846 | 0.981 | 0.544 | 0.859 | 1.000 |

## Result Analysis

CNN + Average Pooling은 가장 단순한 RGB baseline으로, 시간 순서를 직접 모델링하지 않기 때문에 test Macro F1이 0.6246으로 가장 낮았습니다. 특히 파손, 유기, 절도처럼 객체나 상황 맥락이 복잡한 클래스에서 낮은 F1-score를 보였습니다.

CNN + GRU는 같은 RGB 입력을 사용하지만 frame feature sequence의 시간 흐름을 반영하여 test Macro F1이 0.7547로 상승했습니다. 이는 3 FPS 저프레임 영상에서도 temporal modeling이 이상행동 분류에 유효하다는 점을 보여줍니다.

Keypoint 모델은 test Macro F1 0.9497로 가장 높은 성능을 보였습니다. 본 데이터셋의 XML keypoint가 행동 구간의 자세와 움직임을 직접적으로 담고 있기 때문에, 전도, 폭행, 교통약자처럼 사람 자세 변화가 중요한 class에서 특히 강했습니다.

Fusion 모델은 RGB-only 모델보다 높은 test Macro F1 0.8467을 보였지만, keypoint-only 모델보다는 낮았습니다. 현재 fusion은 feature concat 기반의 단순 결합이며 RGB backbone을 freeze한 상태이므로, RGB branch가 keypoint branch보다 충분히 강하게 기여하지 못했을 가능성이 있습니다.

이 한계를 확인하기 위해 추가 모델 `fusion_attention`을 구성했습니다. 이 모델은 keypoint feature를 query로, RGB frame feature를 key/value로 사용하는 cross-attention을 적용합니다. 단순 concatenation이 RGB 노이즈를 그대로 붙이는 방식이라면, cross-attention은 pose feature가 RGB sequence에서 필요한 장면 정보만 선택적으로 참조하도록 설계한 구조입니다. 실험 결과 cross-attention fusion은 기존 concat fusion보다 test Macro F1이 0.8467에서 0.8620으로 상승했습니다. 따라서 단순 결합보다 modality 간 상호작용을 명시적으로 모델링하는 것이 더 유리하다는 방향성을 확인했습니다.

Validation과 test 사이의 차이도 중요합니다. CNN + GRU와 Fusion은 validation 점수에 비해 test 점수가 많이 낮아졌습니다. 이는 AI Hub Training 내부에서 나눈 validation보다 AI Hub Validation 기반 test split이 더 어려운 분포일 수 있음을 의미합니다.

## Training Stability Note

train/val/test 구조로 바꾼 뒤 `baseline_cnn_avg`가 거의 수렴하지 않는 문제가 있었습니다. 원인은 DataLoader shuffle 조건이 예전 split 이름인 `Training`에만 걸려 있고 새 split 이름인 `train`에는 적용되지 않았기 때문입니다. 현재는 `train`과 `Training` 모두 학습 시 shuffle되도록 수정했습니다.

수정 후 `baseline_cnn_avg`를 3 epoch만 확인했을 때 validation Macro F1이 다음처럼 정상적으로 상승했습니다.

```text
epoch 1: valid_f1=0.5292
epoch 2: valid_f1=0.6512
epoch 3: valid_f1=0.6549
```

## Key Files

```text
configs/                         # experiment configs
scripts/extract_aihub_dataset.py  # AI Hub zip extraction
scripts/build_manifest.py         # MP4/XML pair manifest generation
scripts/extract_frames_224.py     # 224x224 frame extraction
scripts/build_train_val_test_manifest.py
scripts/run_all_experiments.py    # sequential train + test evaluation
scripts/evaluate_all_best.py      # evaluate saved best checkpoints
src/data/                         # dataset, XML parser, sampling
src/models/                       # CNN, GRU, keypoint, fusion models
src/train.py                      # common training loop
src/evaluate.py                   # evaluation loop
docs/work_log_ko.md               # Korean work log for report writing
```

## Notes

대용량 데이터와 학습 산출물은 repository에 포함하지 않습니다. 재현하려면 AI Hub 데이터를 같은 구조로 준비한 뒤 전처리 script를 순서대로 실행해야 합니다.

작업 기록과 보고서용 정리는 다음 파일에 누적했습니다.

```text
docs/work_log_ko.md
```
