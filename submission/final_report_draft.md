# CCTV 영상과 사람 자세 정보를 활용한 무인매장 이상행동 분류

## 1. 요약

본 프로젝트는 AI Hub의 실내 편의점/매장 CCTV 이상행동 데이터를 활용하여 무인매장 환경에서 발생할 수 있는 이상행동을 자동 분류하는 딥러닝 모델을 실험한 것이다. 원본 MP4 영상과 XML 라벨링 데이터를 정리하고, 영상은 224x224 RGB 프레임으로 변환했으며, XML에서는 행동 구간과 사람 keypoint 정보를 추출하였다. 1차 실험에서는 RGB 기반 모델, GT keypoint 기반 모델, RGB+GT keypoint fusion 모델을 비교하여 GT keypoint가 이상행동 분류에 매우 강력한 정보임을 확인하였다. 이후 실제 CCTV 추론 환경에서는 GT keypoint가 제공되지 않는다는 한계를 보완하기 위해, 2차 실험에서는 RGB 이미지에서 keypoint를 예측하는 pose estimator를 먼저 학습하고 predicted keypoint 기반 downstream 분류를 수행하였다. 최종적으로 2차 실험에서 RGB-only 모델은 test Macro F1 0.7228, predicted keypoint-only 모델은 0.5980, RGB+predicted keypoint fusion 모델은 0.7270을 기록하여, 예측 keypoint는 단독으로는 부족하지만 RGB feature와 결합할 때 보완적인 정보를 제공할 수 있음을 확인하였다.

## 2. 서론

무인매장과 편의점형 자동화 매장이 증가하면서 CCTV 영상을 활용한 이상행동 자동 인식의 필요성이 커지고 있다. 일반적인 CCTV 영상은 사람이 직접 관찰해야 하므로 실시간 대응이 어렵고, 전도, 폭행, 절도, 방화와 같은 이상행동은 빠르게 감지되어야 한다.

본 프로젝트의 목표는 CCTV 영상에서 다음 8개 행동 class를 분류하는 것이다.

| 번호 | 행동 |
|---:|---|
| 1 | 전도 |
| 2 | 파손 |
| 3 | 방화 |
| 4 | 흡연 |
| 5 | 유기 |
| 6 | 절도 |
| 7 | 폭행 |
| 8 | 이동약자 |

본 프로젝트의 핵심 질문은 다음과 같다.

```text
RGB 영상 정보만 사용하는 것보다 사람 자세 정보(keypoint)를 함께 사용하는 것이 이상행동 분류에 도움이 되는가?
```

이를 확인하기 위해 두 단계의 실험을 설계하였다. 1차 실험에서는 XML에 라벨링된 GT keypoint를 직접 입력으로 사용하여 pose 정보의 잠재력을 확인하였다. 2차 실험에서는 실제 추론 환경을 반영하기 위해 RGB 이미지에서 keypoint를 예측하고, 예측된 keypoint를 downstream 행동 분류에 사용하였다.

## 3. 데이터

### 3.1 데이터 출처

사용 데이터는 AI Hub의 실내 편의점/매장 사람 이상행동 데이터이다.

데이터 설명 페이지:

```text
https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%ED%8E%B8%EC%9D%98%EC%A0%90&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71550
```

AI Hub에서 zip 형태로 데이터를 다운로드한 뒤, 로컬 저장공간 문제를 고려하여 압축 해제 후 원본 zip은 삭제하고 MP4 영상과 XML 라벨 파일만 유지하였다. 실제 실험에는 MP4와 XML이 1:1로 매칭되는 5,841개 clip을 사용하였다.

### 3.2 원본 데이터 구조

원본 데이터는 `Training`과 `Validation`으로 나뉘어 있으며, 각 split 안에 원천데이터 MP4와 라벨링데이터 XML이 포함되어 있었다.

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

압축 해제 및 정리 후에는 다음 구조로 변환하였다.

```text
data/extracted/
|-- Training/
|   |-- videos/
|   `-- labels/
`-- Validation/
    |-- videos/
    `-- labels/
```

데이터 검증 결과는 다음과 같다.

| 항목 | 값 |
|---|---:|
| MP4 파일 수 | 5,841 |
| XML 파일 수 | 5,841 |
| 누락 XML | 0 |
| 영상 FPS | 3 FPS |
| 일반적인 영상 길이 | 약 60초 |
| 일반적인 frame 수 | 180 또는 181 |

### 3.3 전처리 과정

전처리는 `scripts/data_processing/` 아래에 정리하였다.

```text
scripts/data_processing/extract_aihub_dataset.py
scripts/data_processing/build_manifest.py
scripts/data_processing/inspect_video_fps.py
scripts/data_processing/extract_frames_224.py
scripts/data_processing/build_train_val_test_manifest.py
```

전처리 과정은 다음과 같다.

1. AI Hub zip 파일 압축 해제
2. MP4와 XML 파일 구조 정리
3. 영상 FPS, 해상도, frame 수 확인
4. MP4/XML 1:1 매칭 manifest 생성
5. MP4 영상을 224x224 RGB JPEG frame으로 변환
6. XML에서 행동 구간과 17개 관절 keypoint 추출
7. AI Hub Training split을 train/validation으로 재분할
8. AI Hub Validation split은 최종 test set으로 사용

최종 split은 다음과 같다.

| Split | Clip 수 | 설명 |
|---|---:|---|
| Train | 4,154 | AI Hub Training 내부 80% |
| Validation | 1,037 | AI Hub Training 내부 20% |
| Test | 650 | AI Hub Validation 전체 |

RGB frame은 ImageNet pretrained ResNet18 입력과 호환되도록 224x224로 변환하였다. 원본 영상은 대부분 1920x1080 해상도였으나, 이를 그대로 사용하면 GPU memory와 학습 시간이 크게 증가하므로, 본 실험에서는 224x224 frame을 사용하였다.

XML keypoint 좌표는 원본 해상도 기준 pixel 좌표로 저장되어 있었고, 학습 시에는 width와 height로 나누어 0~1 범위로 정규화하였다. 17개 관절의 x, y 좌표를 사용하므로 keypoint 입력 shape은 `[T, 34]`이다.

## 4. 방법

### 4.1 사용한 주요 라이브러리 및 오픈소스

본 프로젝트는 PyTorch 기반으로 구현하였다.

| 도구 | 사용 목적 |
|---|---|
| PyTorch | 모델 구현 및 학습 |
| torchvision | ImageNet pretrained ResNet18 사용 |
| scikit-learn | accuracy, macro F1, confusion matrix 계산 |
| OpenCV | 영상 frame 추출 |
| matplotlib | learning curve, confusion matrix, 결과표 시각화 |

RGB feature extractor는 `torchvision.models.resnet18`의 ImageNet pretrained weight를 사용하였다. 본 프로젝트에서는 이를 CCTV 이상행동 분류에 맞게 feature extractor로 활용하고, 뒤쪽에 GRU 또는 classifier를 연결하였다.

### 4.2 1차 실험: GT keypoint 기반 baseline 비교

1차 실험의 목적은 RGB 영상 정보와 GT keypoint 정보의 분류 성능을 비교하는 것이다. 이 실험에서는 XML에 라벨링된 GT keypoint를 행동 분류 모델 입력으로 직접 사용하였다.

비교 모델은 다음과 같다.

| 모델 | 입력 | 구조 |
|---|---|---|
| CNN + Average Pooling | RGB frame | ResNet18 feature + average pooling |
| CNN + GRU | RGB frame sequence | ResNet18 feature + GRU |
| GT Keypoint 1D-CNN + GRU | XML GT keypoint sequence | 1D-CNN pose encoder + GRU |
| RGB + GT Keypoint Fusion | RGB + XML GT keypoint | RGB branch + pose branch concat |
| RGB + GT Keypoint Cross-Attention | RGB + XML GT keypoint | keypoint query, RGB key/value attention |

모델 구조도:

```text
docs/assets/model_architectures/experiment1_architecture.png
```

### 4.3 2차 실험: predicted keypoint 기반 downstream 비교

1차 실험에서 GT keypoint 기반 모델이 가장 높은 성능을 보였지만, 실제 CCTV 추론 환경에서는 XML GT keypoint가 제공되지 않는다. 따라서 2차 실험에서는 GT keypoint를 분류기에 직접 넣지 않고, RGB 이미지에서 keypoint를 예측하는 pose estimator를 먼저 학습하였다.

2차 실험은 두 단계로 구성된다.

```text
Stage 1: RGB image -> keypoint estimator 학습
Stage 2: RGB only, predicted keypoint only, RGB+predicted keypoint fusion 비교
```

Stage 1에서는 XML GT keypoint를 정답값으로 사용하여 RGB 이미지에서 17개 관절 좌표를 예측하였다. Stage 2에서는 GT keypoint를 사용하지 않고, pose estimator가 예측한 keypoint를 사용하였다.

비교 모델은 다음과 같다.

| 모델 | 입력 | 목적 |
|---|---|---|
| RGB Only | RGB frame sequence | 2차 실험 기준 성능 |
| RGB -> Predicted Keypoint Only | predicted keypoint sequence | 예측 자세 정보만의 성능 확인 |
| RGB + Predicted Keypoint Fusion | RGB + predicted keypoint | RGB와 예측 자세 정보의 보완 가능성 확인 |

모델 구조도:

```text
docs/assets/model_architectures/experiment2_architecture.png
```

### 4.4 제출 코드 구현 설명

제출 코드는 `.py` 파일이 5개 이상이므로 `submission/team1_code_submission.zip`으로 압축하였다. 코드에는 원본 데이터와 학습된 checkpoint는 포함하지 않고, 데이터 가공 코드, 1차 실험 코드, 2차 실험 코드, 시연 영상 생성 코드, 공통 학습/평가 모듈만 포함하였다.

제출 코드의 전체 구조는 다음과 같다.

```text
src/
scripts/data_processing/
scripts/experiment1/
scripts/experiment2/
scripts/demo/
scripts/plot_model_architecture_diagrams.py
scripts/plot_result_tables.py
configs/
requirements.txt
README.md
```

#### 4.4.1 데이터 처리 코드

데이터 처리 코드는 `scripts/data_processing/`와 `src/data/`에 나누어 작성하였다.

| 파일 | 역할 |
|---|---|
| `scripts/data_processing/extract_aihub_dataset.py` | AI Hub zip 파일 압축 해제 및 원본 구조 정리 |
| `scripts/data_processing/build_manifest.py` | MP4와 XML을 1:1로 매칭하여 manifest 생성 |
| `scripts/data_processing/inspect_video_fps.py` | 영상 FPS, 해상도, frame 수 확인 |
| `scripts/data_processing/extract_frames_224.py` | MP4를 224x224 RGB JPEG frame으로 변환 |
| `scripts/data_processing/build_train_val_test_manifest.py` | train/validation/test split manifest 생성 |
| `src/data/xml_parser.py` | XML에서 행동 구간과 keypoint 좌표 파싱 |
| `src/data/frame_sampler.py` | 행동 구간에서 16개 frame을 균등 sampling |
| `src/data/preprocessing.py` | RGB frame resize, tensor 변환, ImageNet normalization |
| `src/data/dataset.py` | PyTorch `Dataset` 구현 |

가장 중요한 부분은 `src/data/dataset.py`의 `AbnormalBehaviorDataset`이다. 이 클래스는 manifest CSV를 읽어 split을 선택하고, 각 clip의 행동 구간(`action_start_frame`, `action_end_frame`)에서 16개 frame을 균등 sampling한다. RGB 모델에는 `frames` tensor를 제공하고, keypoint 모델에는 XML에서 추출한 `keypoints` tensor를 제공한다. 즉 같은 Dataset 클래스를 사용하면서도 `use_frames`, `use_keypoints` 옵션에 따라 RGB-only, keypoint-only, fusion 모델을 모두 학습할 수 있도록 구성하였다.

keypoint 좌표는 XML에 원본 해상도 기준 pixel 값으로 들어 있으므로, `src/data/xml_parser.py`에서 영상 width와 height로 나누어 0~1 범위로 정규화하였다. 따라서 모델 입력은 frame 크기와 무관하게 `[T, 34]` 형태이며, 여기서 `T=16`, `34=17개 관절 x/y 좌표`이다.

#### 4.4.2 공통 학습 및 평가 코드

1차 실험의 공통 학습 루프는 `src/train.py`에 구현하였다.

| 함수 | 설명 |
|---|---|
| `model_modalities()` | 모델 이름에 따라 RGB 입력과 keypoint 입력 필요 여부 결정 |
| `make_loader()` | split별 `DataLoader` 생성, train split에만 shuffle 적용 |
| `forward_batch()` | batch에서 frames/keypoints를 꺼내 GPU로 이동한 뒤 모델 forward 수행 |
| `run_epoch()` | train 또는 validation 1 epoch 수행 |
| `classification_metrics()` | accuracy, class별 precision/recall/F1, macro F1 계산 |
| `train_one()` | config 하나에 대해 전체 학습, best checkpoint 저장, learning curve 저장 |

모델 선택은 validation Macro F1 기준으로 수행하였다. `train_one()`은 매 epoch마다 train/validation loss와 Macro F1을 출력하고, validation Macro F1이 가장 높을 때 `outputs/.../checkpoints/`에 best checkpoint를 저장한다. 이후 `src/evaluate.py`의 `evaluate_checkpoint()`가 best checkpoint를 불러와 AI Hub Validation 기반 test split에서 최종 성능을 계산한다. 이렇게 validation은 모델 선택에만 사용하고, test는 최종 평가에만 사용하도록 분리하였다.

#### 4.4.3 1차 실험 코드

1차 실험 실행 코드는 `scripts/experiment1/run_experiment1.py`이다. 이 스크립트는 다음 config들을 순서대로 읽어 같은 방식으로 학습한다.

```text
scripts/experiment1/configs/baseline_cnn_avg.yaml
scripts/experiment1/configs/baseline_cnn_lstm.yaml
scripts/experiment1/configs/baseline_keypoint.yaml
scripts/experiment1/configs/fusion.yaml
```

각 config에는 모델 이름, batch size, epoch 수, learning rate, split 이름, frame 수 등이 들어 있다. `run_experiment1.py`는 config를 읽고 `src/train.py`의 `train_one()`을 호출한 뒤, best checkpoint를 test split에서 평가한다. 따라서 1차 실험의 모든 모델은 동일한 split, 동일한 metric, 동일한 best model selection 규칙으로 비교된다.

1차 실험 모델 구현은 `src/models/`에 있다.

| 파일 | 모델 | 핵심 구현 |
|---|---|---|
| `src/models/cnn_avg.py` | CNN + Average Pooling | ResNet18 feature를 frame별 추출 후 평균 pooling |
| `src/models/cnn_lstm.py` | CNN + GRU | ResNet18 feature sequence를 GRU로 temporal modeling |
| `src/models/keypoint_lstm.py` | GT Keypoint 1D-CNN + GRU | keypoint sequence를 1D-CNN과 GRU로 분류 |
| `src/models/fusion.py` | RGB + GT Keypoint Fusion | RGB branch와 pose branch feature concat |
| `src/models/fusion.py` | RGB + GT Keypoint Cross-Attention | pose token을 query로, RGB token을 key/value로 사용하는 attention fusion |
| `src/models/build.py` | 모델 factory | config의 `model` 값에 따라 모델 객체 생성 |

RGB branch는 `torchvision.models.resnet18`의 ImageNet pretrained weight를 사용한다. 본 실험에서는 backbone을 feature extractor로 사용하기 위해 기본적으로 freeze하였고, GRU와 classifier head만 학습하였다. keypoint branch는 34차원 좌표 sequence를 `Conv1d -> BatchNorm -> ReLU -> GRU`로 처리한다.

Fusion 모델은 RGB branch와 keypoint branch를 각각 encoding한 뒤 마지막 hidden state를 concat하여 class를 예측한다. Cross-Attention Fusion은 단순 concat의 한계를 줄이기 위해 pose token을 query로 두고 RGB token을 key/value로 두어, 자세 정보가 RGB sequence에서 필요한 시각 정보를 선택적으로 참조하도록 구현하였다.

#### 4.4.4 2차 실험 코드

2차 실험 코드는 `scripts/experiment2/run_experiment2.py` 하나로 실행되도록 구성하였다. 이 파일은 보고서 실험 흐름과 동일하게 두 stage로 나뉜다.

```text
Stage 1: ImageKeypointEstimator 학습
Stage 2: RGB Only / Predicted Keypoint Only / RGB + Predicted Keypoint Fusion 학습 및 평가
```

`ImageKeypointEstimator`는 RGB frame을 ResNet18 encoder에 통과시킨 뒤, 17개 관절의 x/y 좌표와 visibility logit을 예측한다. 좌표 출력은 sigmoid를 적용하여 0~1 범위로 제한하였다. loss는 좌표 오차와 visibility loss를 함께 사용한다.

```text
pose loss = coordinate loss + 0.1 * visibility loss
```

좌표 성능은 normalized MPJPE로 측정하였다. MPJPE는 예측 keypoint와 GT keypoint 사이의 평균 거리이며 낮을수록 좋다. visibility는 GT keypoint가 존재하는 관절인지 여부를 보조적으로 학습하기 위해 사용하였다.

Stage 2에서는 pose estimator를 고정한 상태로 predicted keypoint를 생성하고, 다음 세 모델을 비교한다.

| 모델 | 코드 클래스 | 입력 |
|---|---|---|
| RGB Only | `RGBSequenceClassifier` | RGB frame sequence |
| Predicted Keypoint Only | `KeypointSequenceClassifier` | pose estimator가 예측한 keypoint |
| RGB + Predicted Keypoint Fusion | `RGBPredictedKeypointFusionClassifier` | RGB frame sequence + predicted keypoint |

`run_classifier_epoch()`는 mode 값에 따라 입력을 다르게 구성한다. `rgb_only`는 RGB frame만 classifier에 전달하고, `pred_keypoint_only`와 `pred_keypoint_fusion`은 먼저 pose estimator로 predicted keypoint를 만든 뒤 classifier에 전달한다. 이때 downstream 분류기에는 XML GT keypoint를 직접 넣지 않으므로, 실제 CCTV 추론 환경에 더 가까운 평가가 된다.

2차 실험도 validation Macro F1 기준으로 best checkpoint를 저장하고, 마지막에는 test split에서 최종 성능을 계산한다. 결과는 `outputs/experiment2/metrics/`, learning curve와 confusion matrix는 `outputs/experiment2/figures/`에 저장된다.

#### 4.4.5 시연 영상 코드

시연 영상 생성 코드는 `scripts/demo/make_realtime_cctv_demo.py`이다. 이 스크립트는 test split의 실제 CCTV clip을 읽고, 2차 실험의 `ImageKeypointEstimator`와 `RGBPredictedKeypointFusionClassifier` best checkpoint를 불러와 예측 결과를 영상 위에 표시한다.

주요 기능은 다음과 같다.

| 함수 | 설명 |
|---|---|
| `choose_sample()` | manifest에서 test clip 선택 |
| `predict_clip()` | 16개 action frame으로 class probability와 clip inference time 계산 |
| `predict_frame_keypoints()` | frame-wise predicted keypoint 생성 및 pose inference time 측정 |
| `draw_skeleton()` | predicted keypoint를 CCTV 화면 위에 skeleton으로 표시 |
| `draw_panel()` | prediction, confidence, top-3 probability, frame progress overlay |

현재 pose estimator는 사람 detector가 아니므로 사람이 명확하지 않은 구간에서도 좌표를 출력할 수 있다. 따라서 제출용 데모는 `--pose-display-mode action`을 기본값으로 사용하여 fall alert 직전/행동 구간에서만 predicted keypoint를 표시한다. `--pose-display-mode always`를 사용하면 raw predicted keypoint를 모든 frame에 표시할 수 있고, `--pose-display-mode off`를 사용하면 skeleton overlay를 끌 수 있다.

또한 데모 스크립트는 CUDA 동기화 후 시간을 측정하여 `cctv_realtime_demo_timing.json`에 저장한다. 이를 통해 model-only 시간과 video decoding/preprocessing을 포함한 end-to-end 시간을 구분하여 실제 적용 가능성을 분석하였다.

## 5. 실험 결과

### 5.1 1차 실험 결과

1차 실험 결과는 다음과 같다.

| 모델 | Best Epoch | Valid Macro F1 | Test Accuracy | Test Macro F1 |
|---|---:|---:|---:|---:|
| CNN + Average Pooling | 19 | 0.8263 | 0.6323 | 0.6246 |
| CNN + GRU | 17 | 0.9611 | 0.7554 | 0.7547 |
| GT Keypoint 1D-CNN + GRU | 30 | 0.9864 | 0.9492 | 0.9497 |
| RGB + GT Keypoint Fusion | 21 | 0.9805 | 0.8523 | 0.8467 |
| RGB + GT Keypoint Cross-Attention | 27 | 0.9826 | 0.8646 | 0.8620 |

결과표 이미지:

```text
docs/assets/result_tables/experiment1_results_table.png
```

1차 실험에서 가장 높은 성능은 GT Keypoint 1D-CNN + GRU 모델의 test Macro F1 0.9497이었다. 이는 사람의 자세 정보가 이상행동 분류에 매우 강력한 특징임을 보여준다. 특히 전도, 폭행, 이동약자처럼 사람 자세와 움직임 변화가 중요한 class에서 keypoint 정보가 유리하게 작용하였다.

그러나 이 결과는 실제 추론 환경과 차이가 있다. 실제 CCTV 영상에는 XML GT keypoint가 제공되지 않으므로, 1차 실험 결과는 pose 정보가 충분히 정확할 때의 상한 성능에 가깝게 해석해야 한다.

### 5.2 2차 실험 결과

2차 실험에서 pose estimator의 best validation normalized MPJPE는 0.0648이었다. MPJPE는 Mean Per Joint Position Error로, 예측 관절 좌표와 GT 관절 좌표 사이의 평균 거리이며 낮을수록 좋다.

2차 실험 downstream 분류 결과는 다음과 같다.

| 모델 | Best Epoch | Valid Macro F1 | Test Accuracy | Test Macro F1 |
|---|---:|---:|---:|---:|
| RGB Only | 13 | 0.9264 | 0.7231 | 0.7228 |
| RGB -> Predicted Keypoint Only | 13 | 0.7605 | 0.5908 | 0.5980 |
| RGB + Predicted Keypoint Fusion | 15 | 0.9450 | 0.7246 | 0.7270 |

결과표 이미지:

```text
docs/assets/result_tables/experiment2_results_table.png
```

Predicted keypoint-only 모델은 RGB-only보다 낮은 성능을 보였다. 이는 이미지에서 예측한 keypoint가 GT keypoint만큼 안정적이지 않고, pose estimation 오차가 downstream 분류 성능에 누적되었기 때문으로 해석된다.

반면 RGB + predicted keypoint fusion 모델은 RGB-only보다 test Macro F1이 0.7228에서 0.7270으로 소폭 상승하였다. 개선 폭은 크지 않지만, predicted keypoint가 RGB feature와 결합될 때 자세 정보를 보완적으로 제공할 가능성을 확인하였다.

### 5.3 Validation과 Test 차이

2차 실험에서는 validation 성능과 test 성능 사이에 차이가 있었다. 예를 들어 RGB + predicted keypoint fusion 모델은 validation Macro F1 0.9450을 기록했지만 test Macro F1은 0.7270이었다. 이는 AI Hub Training 내부에서 나눈 validation split보다 AI Hub Validation 기반 test split이 더 어려운 분포를 가질 수 있음을 의미한다. 따라서 최종 성능 해석은 test set 기준으로 수행하였다.

## 6. 토의 내용

### 6.1 프로젝트 수행 과정에서 배운 점

본 프로젝트를 통해 단순히 모델 구조를 복잡하게 만드는 것보다 데이터 split, GT 정보 사용 여부, 실제 추론 환경과의 일치 여부가 성능 해석에서 매우 중요하다는 점을 확인하였다. 1차 실험에서 GT keypoint-only 모델은 매우 높은 성능을 보였지만, 이는 실제 배포 환경에서는 사용할 수 없는 정보였다. 따라서 2차 실험처럼 GT keypoint를 pose estimator 학습에만 사용하고 downstream에서는 predicted keypoint를 사용하는 구조가 더 현실적인 평가 방식임을 알 수 있었다.

또한 validation 성능이 높더라도 test set에서 성능이 크게 달라질 수 있음을 확인하였다. 이 때문에 모델 선택은 validation 기준으로 하되, 최종 결론은 test 결과를 중심으로 해석해야 한다.

### 6.2 팀원 역할 분담

아래 내용은 팀원 이름에 맞게 수정하여 제출한다.

| 팀원 | 담당 역할 |
|---|---|
| [팀원 1] | AI Hub 데이터 다운로드, 압축 해제, 데이터 구조 정리 |
| [팀원 2] | MP4 frame 추출, XML keypoint 파싱, train/validation/test split 구성 |
| [팀원 3] | 1차 실험 모델 구현 및 학습 |
| [팀원 4] | 2차 실험 pose estimator 및 fusion 모델 구현 |
| [팀원 5] | 결과 분석, README/보고서/PPT 정리 |

### 6.3 협업 및 공유 방식

프로젝트 코드는 GitHub repository를 통해 공유하였다.

```text
https://github.com/kiseo00choi-bit/multimodal_project_team1.git
```

실험 결과와 작업 기록은 `README.md`, `docs/work_log_ko.md`, `outputs/experiment1`, `outputs/experiment2`에 정리하였다. 데이터 크기가 매우 크기 때문에 원본 데이터와 frame 데이터는 GitHub에 올리지 않고, 코드와 결과 분석 파일 중심으로 공유하였다.

제출 파일에서도 1차 실험과 2차 실험을 분리하였다. 1차 실험 요약은 `submission/experiment1/README.md`와 `submission/experiment1/experiment1_results_table.png`에 정리했고, 2차 실험 요약은 `submission/experiment2/README.md`와 `submission/experiment2/experiment2_results_table.png`에 정리하였다. 코드 구조도 동일하게 `scripts/experiment1/`은 GT keypoint 기반 baseline 비교, `scripts/experiment2/`는 RGB에서 predicted keypoint를 추정한 뒤 downstream 분류를 수행하는 실험으로 분리하였다.

시연 영상은 `scripts/demo/make_realtime_cctv_demo.py`로 생성하였다. 해당 스크립트는 AI Hub test split의 실제 CCTV clip을 읽고, 2차 실험의 `RGB + Predicted Keypoint Fusion` best checkpoint를 사용하여 예측 class와 confidence를 영상 위에 overlay한다. 또한 pose estimator가 각 frame에서 예측한 predicted keypoint skeleton을 함께 표시하여, 2차 실험에서 RGB 영상으로부터 추정한 자세 정보가 downstream fusion 분류에 사용되는 과정을 시각적으로 확인할 수 있도록 하였다. 생성된 제출용 영상은 `submission/cctv_realtime_demo.mp4`이다. 영상에는 `LIVE CCTV MONITOR`, 예측 class, confidence, GT label, top-3 probability, predicted keypoint skeleton, frame progress가 표시되어 실제 CCTV 관제 화면처럼 모델 추론 결과를 확인할 수 있도록 구성하였다.

시연 영상의 skeleton은 XML GT keypoint가 아니라 RGB frame만 입력받은 pose estimator의 실제 예측 결과이다. 따라서 사람 detector 없이 전체 CCTV frame에서 관절 좌표를 직접 회귀하는 현재 구조에서는 사람이 화면에서 작게 보이거나 배경 물체와 겹치는 구간에서 keypoint가 사람 위치와 어긋날 수 있다. 특히 사람이 명확히 보이지 않는 구간에서도 회귀 모델은 항상 17개 좌표를 출력하므로, raw 결과를 모든 frame에 표시하면 빈 공간에 skeleton이 뜨는 문제가 발생할 수 있다. 이 현상은 시각화 오류라기보다 현재 pose estimator에 사람 존재 여부와 keypoint confidence가 없다는 한계로 해석하는 것이 적절하다. 최종 제출용 시연에서는 불안정한 raw pose를 상시 표시하지 않고, fall alert 직전/행동 구간에서만 predicted keypoint를 표시하는 `pose_display_mode=action`을 사용하였다. 연구용으로 raw pose 결과를 확인할 때는 `pose_display_mode=always`를 사용할 수 있다.

RTX 5070 Ti 환경에서 추론 시간을 측정한 결과, frame-wise pose estimator의 model-only 시간은 약 0.27 ms/frame, 영상 decode와 preprocessing을 포함한 end-to-end pose 추론은 약 9.24 ms/frame이었다. 16 frame clip에 대해 pose estimator와 RGB+predicted-keypoint fusion classifier를 함께 실행한 clip 단위 추론 시간은 약 179.19 ms/clip이었다. 이는 16 frame 기준 약 89.29 frame/s에 해당한다. 본 데이터의 원본 CCTV는 약 3 FPS 수준이므로 단일 CCTV stream 기준으로는 실시간 적용 가능성이 있다고 볼 수 있다. 다만 실제 매장 환경에서 여러 CCTV stream을 동시에 처리하려면 batching, 영상 decode 최적화, 더 안정적인 사람 검출 기반 pose estimator가 필요하다.

### 6.4 한계 및 향후 개선 방향

현재 2차 실험의 fusion 개선 폭은 크지 않다. 이는 predicted keypoint의 품질이 GT keypoint만큼 안정적이지 않고, 단순 concat fusion이 keypoint 예측 오차를 충분히 제어하지 못했기 때문일 수 있다. 특히 시연 영상에서 확인한 것처럼 predicted keypoint가 실제 사람 위치와 어긋나는 경우 downstream classifier에는 자세 정보가 아니라 노이즈가 추가될 수 있다.

향후 개선 방향은 다음과 같다.

1. pose estimator를 더 강한 구조로 개선한다.
2. ResNet18 backbone 일부를 unfreeze하여 CCTV 도메인에 fine-tuning한다.
3. YOLO 계열 사람 검출기 또는 OpenPose/HRNet 계열 pose estimator처럼 사람 영역을 먼저 찾는 구조를 검토한다.
4. keypoint confidence 또는 visibility를 fusion weight에 반영한다.
5. 단순 concat 대신 cross-attention 또는 gating 기반 fusion을 적용한다.
6. pose estimator와 action classifier를 end-to-end로 fine-tuning한다.

## 7. 결론

본 프로젝트에서는 CCTV 이상행동 분류에서 RGB 영상 정보와 사람 자세 정보의 역할을 비교하였다. 1차 실험에서는 GT keypoint 기반 모델이 가장 높은 성능을 보여, 자세 정보가 이상행동 분류에 매우 중요한 특징임을 확인하였다. 그러나 실제 CCTV 추론 환경에서는 GT keypoint가 제공되지 않기 때문에, 2차 실험에서는 RGB 이미지에서 keypoint를 예측한 뒤 downstream 분류에 사용하는 구조를 설계하였다.

2차 실험 결과 predicted keypoint-only 모델은 RGB-only보다 낮았지만, RGB + predicted keypoint fusion 모델은 RGB-only보다 test Macro F1을 소폭 개선하였다. 따라서 사람 자세 정보는 이상행동 분류에 유효하지만, 실제 적용에서는 keypoint 예측 품질이 전체 성능의 중요한 병목임을 확인하였다.
