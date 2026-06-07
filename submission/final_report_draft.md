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

시연 영상은 `scripts/demo/make_realtime_cctv_demo.py`로 생성하였다. 해당 스크립트는 AI Hub test split의 실제 CCTV clip을 읽고, 2차 실험의 `RGB + Predicted Keypoint Fusion` best checkpoint를 사용하여 예측 class와 confidence를 영상 위에 overlay한다. 또한 pose estimator가 각 frame에서 예측한 predicted keypoint skeleton을 함께 표시하여, 2차 실험에서 RGB 영상으로부터 추정한 자세 정보가 downstream fusion 분류에 사용되는 과정을 시각적으로 확인할 수 있도록 하였다. 생성된 제출용 영상은 `submission/cctv_realtime_demo.mp4`이며 약 30초 길이이다. 영상에는 `LIVE CCTV MONITOR`, 예측 class, confidence, GT label, top-3 probability, predicted keypoint skeleton, frame progress가 표시되어 실제 CCTV 관제 화면처럼 모델 추론 결과를 확인할 수 있도록 구성하였다.

### 6.4 한계 및 향후 개선 방향

현재 2차 실험의 fusion 개선 폭은 크지 않다. 이는 predicted keypoint의 품질이 GT keypoint만큼 안정적이지 않고, 단순 concat fusion이 keypoint 예측 오차를 충분히 제어하지 못했기 때문일 수 있다.

향후 개선 방향은 다음과 같다.

1. pose estimator를 더 강한 구조로 개선한다.
2. ResNet18 backbone 일부를 unfreeze하여 CCTV 도메인에 fine-tuning한다.
3. keypoint confidence 또는 visibility를 fusion weight에 반영한다.
4. 단순 concat 대신 cross-attention 또는 gating 기반 fusion을 적용한다.
5. pose estimator와 action classifier를 end-to-end로 fine-tuning한다.

## 7. 결론

본 프로젝트에서는 CCTV 이상행동 분류에서 RGB 영상 정보와 사람 자세 정보의 역할을 비교하였다. 1차 실험에서는 GT keypoint 기반 모델이 가장 높은 성능을 보여, 자세 정보가 이상행동 분류에 매우 중요한 특징임을 확인하였다. 그러나 실제 CCTV 추론 환경에서는 GT keypoint가 제공되지 않기 때문에, 2차 실험에서는 RGB 이미지에서 keypoint를 예측한 뒤 downstream 분류에 사용하는 구조를 설계하였다.

2차 실험 결과 predicted keypoint-only 모델은 RGB-only보다 낮았지만, RGB + predicted keypoint fusion 모델은 RGB-only보다 test Macro F1을 소폭 개선하였다. 따라서 사람 자세 정보는 이상행동 분류에 유효하지만, 실제 적용에서는 keypoint 예측 품질이 전체 성능의 중요한 병목임을 확인하였다.
