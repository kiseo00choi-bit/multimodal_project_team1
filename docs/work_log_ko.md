# 프로젝트 작업 기록

이 문서는 딥러닝 응용 팀 프로젝트 수행 과정에서 Codex와 함께 진행한 초기 환경 구성, 데이터 정리, 데이터 검증 내용을 보고서 작성에 활용할 수 있도록 정리한 기록입니다.

## 1. 프로젝트 개요

본 프로젝트는 실내 무인 매장 CCTV 영상과 XML 라벨링 데이터를 활용하여 이상행동을 분류하는 딥러닝 모델을 구현하고 비교하는 것을 목표로 합니다.

비교 대상 모델은 다음과 같습니다.

- CNN + Average Pooling 기반 RGB 영상 모델
- CNN + LSTM 기반 RGB 영상 모델
- 1D-CNN + LSTM 기반 keypoint 모델
- RGB 영상 특징과 keypoint 특징을 결합한 fusion 모델

분류 대상 이상행동 클래스는 다음 8개입니다.

- 07.전도
- 08.파손
- 09.방화
- 10.흡연
- 11.유기
- 12.절도
- 13.폭행
- 14.교통약자

## 2. 작업 환경 구성

프로젝트 작업 경로는 다음과 같습니다.

```text
C:\Users\rltjr\Desktop\아주대학교\4학년\딥러닝응용\team_project
```

사용 GPU는 NVIDIA GeForce RTX 5070 Ti입니다.

가상환경은 프로젝트 폴더 내부에 `.venv5070` 이름으로 생성했습니다.

```powershell
python -m venv .venv5070
```

가상환경 활성화 명령은 다음과 같습니다.

```powershell
.\.venv5070\Scripts\Activate.ps1
```

PyTorch는 RTX 5070 Ti에서 CUDA를 사용할 수 있도록 CUDA 12.8 wheel을 설치했습니다.

설치 및 확인 결과는 다음과 같습니다.

```text
torch version: 2.11.0+cu128
CUDA version: 12.8
torch.cuda.is_available(): True
GPU name: NVIDIA GeForce RTX 5070 Ti
```

주요 설치 패키지는 다음과 같습니다.

- torch
- torchvision
- torchaudio
- numpy
- pandas
- scikit-learn
- opencv-python
- pillow
- pyyaml
- tqdm
- matplotlib
- seaborn
- tensorboard
- rich

## 3. 프로젝트 폴더 구조 정리

초기에는 `codex.md`에 제안된 구조를 기준으로 프로젝트 폴더를 구성했습니다. 이후 실제 데이터 경로와 실험 흐름에 맞게 다음과 같은 구조로 정리했습니다.

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
|   |-- processed/
|   `-- splits/
|-- docs/
|   `-- work_log_ko.md
|-- notebooks/
|-- outputs/
|   |-- checkpoints/
|   |-- figures/
|   |-- logs/
|   `-- metrics/
|-- scripts/
|   |-- build_manifest.py
|   |-- extract_aihub_dataset.py
|   `-- inspect_video_fps.py
`-- src/
    |-- data/
    |-- models/
    |-- train.py
    |-- evaluate.py
    `-- utils.py
```

대용량 데이터는 Git에 포함하지 않도록 `.gitignore`에 제외 규칙을 추가했습니다.

```text
data/raw/
data/extracted/
data/processed/
data/splits/
outputs/
.venv*/
```

## 4. 원본 데이터 이동 및 정리

AI Hub에서 다운로드한 원본 데이터는 처음에 다음 경로에 있었습니다.

```text
C:\Users\rltjr\Downloads\238-2.실내(편의점, 매장) 사람 이상행동 데이터
```

데이터 용량이 매우 크기 때문에 복사하지 않고 `Move-Item`을 사용하여 프로젝트 경로로 이동했습니다. 이후 코드에서 다루기 쉽게 긴 한글 폴더명을 `data/raw`로 정리했습니다.

원본 zip 데이터 구조는 다음과 같았습니다.

```text
data/raw/01-1.정식개방데이터/
|-- Training/
|   |-- 01.원천데이터/
|   `-- 02.라벨링데이터/
`-- Validation/
    |-- 01.원천데이터/
    `-- 02.라벨링데이터/
```

AI Hub 데이터는 이미 `Training`과 `Validation`으로 분리되어 있었으므로, 원본 제공 split을 유지하는 방향으로 결정했습니다. 이는 실험 재현성과 데이터 누수 방지 측면에서 적절합니다.

## 5. 압축 파일 상태 확인

처음 압축도구로 zip 파일을 열었을 때 내부가 비어 보이는 문제가 있었습니다. 그러나 Python의 `zipfile` 모듈로 확인한 결과 zip 파일은 정상으로 확인되었습니다.

예시 확인 결과는 다음과 같습니다.

```text
파일: TS_03.이상행동_07.전도.zip
크기: 28,540,342,015 bytes
zipfile.is_zipfile: True
내부 파일 수: 645개
첫 파일: /C_3_7_10_BU_DYA_08-23_13-47-40_CC_RGB_DF2_M2.mp4
```

압축도구에서 비어 보인 원인은 zip 내부 파일 경로가 `/파일명.mp4`처럼 절대 경로 형태에 가깝게 저장되어 있어 일부 압축 프로그램에서 목록 표시가 정상적으로 되지 않았기 때문으로 판단했습니다.

## 6. 압축 해제 및 데이터셋 재구성

전체 zip 파일은 32개였고, 압축 파일 합계는 약 253.17GB였습니다. 압축 해제 전 C 드라이브 여유 공간은 약 385GB였으며, zip 내부의 실제 압축 해제 예상 용량은 약 255.63GB였습니다.

압축 해제를 위해 `scripts/extract_aihub_dataset.py`를 작성했습니다. 이 스크립트는 zip 내부 파일명의 앞쪽 `/`를 제거하고, split, 데이터 종류, 클래스별로 파일을 정리합니다.

압축 해제 후 데이터 구조는 다음과 같습니다.

```text
data/extracted/
|-- Training/
|   |-- videos/
|   |   |-- 07.전도/
|   |   |-- 08.파손/
|   |   |-- 09.방화/
|   |   |-- 10.흡연/
|   |   |-- 11.유기/
|   |   |-- 12.절도/
|   |   |-- 13.폭행/
|   |   `-- 14.교통약자/
|   `-- labels/
|       |-- 07.전도/
|       |-- 08.파손/
|       |-- 09.방화/
|       |-- 10.흡연/
|       |-- 11.유기/
|       |-- 12.절도/
|       |-- 13.폭행/
|       `-- 14.교통약자/
`-- Validation/
    |-- videos/
    `-- labels/
```

압축 해제 결과는 다음과 같습니다.

```text
압축 해제 파일 수: 11,682개
MP4 파일 수: 5,841개
XML 파일 수: 5,841개
압축 해제 후 data/extracted 용량: 255.63GB
```

압축 해제본이 정상적으로 만들어진 후, 중복 저장을 피하기 위해 원본 zip 폴더인 `data/raw`는 삭제했습니다.

삭제 후 확인 결과는 다음과 같습니다.

```text
data/raw 존재 여부: False
C 드라이브 여유 공간: 약 382.6GB
```

## 7. 영상-라벨 매칭 manifest 생성

학습 코드에서 영상과 XML 라벨을 안정적으로 매칭하기 위해 `scripts/build_manifest.py`를 작성했습니다.

생성된 파일은 다음과 같습니다.

```text
data/splits/manifest.csv
```

manifest에는 다음 컬럼을 저장했습니다.

- split
- label
- video_path
- xml_path
- has_xml

생성 결과는 다음과 같습니다.

```text
samples: 5,841개
missing_xml: 0개
```

클래스별 샘플 수는 다음과 같습니다.

```text
Training
07.전도: 645
08.파손: 642
09.방화: 651
10.흡연: 686
11.유기: 642
12.절도: 641
13.폭행: 641
14.교통약자: 643

Validation
07.전도: 81
08.파손: 80
09.방화: 82
10.흡연: 86
11.유기: 80
12.절도: 80
13.폭행: 80
14.교통약자: 81
```

전체적으로 클래스별 샘플 수는 비교적 균형 잡힌 편입니다.

### 학습 레이블 매핑

모델 학습에서는 PyTorch `CrossEntropyLoss`를 사용하므로 클래스 레이블을 정수형 `class_id`로 변환했습니다. 코드 내부의 `class_id`는 0부터 시작합니다.

| 모델 class_id | 보고서용 번호 | AI Hub class code | 영문 label | 한글 label |
|---:|---:|---|---|---|
| 0 | 1 | 07 | fall | 전도 |
| 1 | 2 | 08 | broken | 파손 |
| 2 | 3 | 09 | fire | 방화 |
| 3 | 4 | 10 | smoke | 흡연 |
| 4 | 5 | 11 | abandon | 유기 |
| 5 | 6 | 12 | theft | 절도 |
| 6 | 7 | 13 | fight | 폭행 |
| 7 | 8 | 14 | weak_pedestrian | 교통약자 |

즉, 학습 코드 기준으로는 `class_id=5`가 절도이고 `class_id=6`이 폭행입니다. 보고서에서 사람이 읽기 쉽게 1번부터 번호를 붙이면 6번이 절도, 7번이 폭행에 해당합니다.

이 매핑은 다음 파일에도 저장되어 있습니다.

```text
data/processed/label_map.json
```

## 8. 영상 FPS 및 XML 프레임 정보 확인

동영상 뷰어에서 영상이 저프레임처럼 보였기 때문에 OpenCV를 사용하여 전체 영상의 FPS와 프레임 수를 확인했습니다. 이를 위해 `scripts/inspect_video_fps.py`를 작성했습니다.

전체 5,841개 MP4 영상에 대한 확인 결과는 다음과 같습니다.

```text
전체 영상 FPS: 3.0 FPS
전체 영상 수: 5,841개
프레임 수 범위: 180~190프레임
대부분 영상 길이: 약 60초
```

FPS 분포는 다음과 같습니다.

```text
3.0 FPS: 5,841개
```

즉, 모든 영상은 3 FPS로 저장된 저프레임 CCTV 영상입니다. 따라서 뷰어에서 프레임이 낮아 보인 것은 정상적인 데이터 특성으로 판단했습니다.

XML 파일의 `<meta><task>` 영역에는 `size`, `start_frame`, `stop_frame`이 포함되어 있으며, 대부분 영상 프레임 수와 일치했습니다.

XML size 분포는 다음과 같습니다.

```text
180프레임: 4,370개
181프레임: 1,459개
182프레임: 6개
189프레임: 5개
190프레임: 1개
```

영상 프레임 수와 XML 프레임 수가 다른 경우는 전체 중 1개였습니다.

```text
파일:
data/extracted/Validation/videos/10.흡연/C_3_10_37_BU_SMB_09-02_17-05-41_CC_RGB_DF2_F3.mp4

video_frames: 180
xml_size: 181
xml_range: 0-180
```

이 1개 파일은 학습 데이터 구성 시 별도 예외 처리가 필요합니다. 예를 들어 실제 영상 프레임 수를 기준으로 XML frame index를 clamp하거나, 마지막 XML frame을 무시하는 방식으로 처리할 수 있습니다.

## 9. 현재 설정 파일 반영 내용

모든 실험 config 파일은 압축 해제된 데이터와 manifest를 사용하도록 수정했습니다.

```yaml
dataset_root: "data/extracted"
manifest_path: "data/splits/manifest.csv"
```

적용된 config 파일은 다음과 같습니다.

- `configs/baseline_cnn_avg.yaml`
- `configs/baseline_cnn_lstm.yaml`
- `configs/baseline_keypoint.yaml`
- `configs/fusion.yaml`

## 10. 현재까지의 결론

현재 데이터셋은 압축 해제 및 정리가 완료되었으며, 영상과 XML 라벨 파일이 모두 1:1로 매칭됩니다. 전체 영상은 3 FPS의 저프레임 CCTV 데이터이며, 대부분 약 60초 길이와 180~181프레임을 가집니다.

따라서 모델 입력을 구성할 때는 전체 영상을 그대로 사용하는 대신, XML의 action segment 또는 전체 frame range에서 `num_frames=16` 또는 `num_frames=32`로 uniform sampling하는 방식이 적절합니다. XML frame index는 대체로 영상 frame index와 직접 대응한다고 볼 수 있으나, 일부 파일의 1 frame 차이는 예외 처리해야 합니다.

다음 단계는 XML 구조를 상세히 분석하여 class label, bounding box, keypoint 좌표를 파싱하는 `src/data/xml_parser.py`를 구현하는 것입니다.

## 11. CNN/LSTM 학습용 프레임 이미지 전처리

전체 영상이 3 FPS임을 확인한 뒤, CNN 및 CNN-LSTM 계열 모델에서 바로 사용할 수 있도록 모든 MP4 영상을 프레임 이미지로 변환했습니다.

전처리 스크립트는 다음과 같습니다.

```text
scripts/extract_frames_224.py
```

프레임 저장 구조는 다음과 같습니다.

```text
data/processed/frames_224/
|-- Training/
|   |-- 07_fall/
|   |-- 08_broken/
|   |-- 09_fire/
|   |-- 10_smoke/
|   |-- 11_abandon/
|   |-- 12_theft/
|   |-- 13_fight/
|   `-- 14_weak_pedestrian/
`-- Validation/
    |-- 07_fall/
    |-- 08_broken/
    |-- 09_fire/
    |-- 10_smoke/
    |-- 11_abandon/
    |-- 12_theft/
    |-- 13_fight/
    `-- 14_weak_pedestrian/
```

각 영상은 별도 폴더로 분리했으며, 프레임 파일명은 다음 형식으로 저장했습니다.

```text
frame_000000.jpg
frame_000001.jpg
...
```

이미지는 CNN 입력에 바로 사용할 수 있도록 `224 x 224` RGB JPEG로 저장했습니다.

프레임 전처리 결과는 다음과 같습니다.

```text
영상 수: 5,841개
저장된 JPEG 프레임 수: 1,052,905개
프레임 이미지 크기: 224 x 224
프레임 디렉터리 불일치: 0개
전처리 데이터 용량: 약 19.65GB
```

영상 단위 및 프레임 단위 매칭을 위해 다음 파일을 생성했습니다.

```text
data/processed/frames_224_manifest.csv
data/processed/frame_index_224.csv
data/processed/label_map.json
```

`frames_224_manifest.csv`에는 영상 단위의 메타데이터를 저장했습니다.

- split
- class_id
- class_code
- label_en
- label_ko
- video_stem
- video_path
- xml_path
- frame_dir
- fps
- source_width
- source_height
- image_size
- frame_count
- xml_size
- xml_start_frame
- xml_stop_frame
- action_start_frame
- action_end_frame
- action_frame_count

`frame_index_224.csv`에는 프레임 단위의 정보를 저장했습니다.

- split
- class_id
- class_code
- label_en
- video_stem
- frame_index
- image_path
- xml_path
- in_action_segment

특히 `action_start_frame`, `action_end_frame`, `in_action_segment`를 함께 저장하여, 추후 전체 영상 기반 학습과 action segment 중심 학습을 모두 실험할 수 있도록 구성했습니다.

현재 상태에서 RGB 기반 CNN 및 CNN-LSTM 학습을 시작할 수 있습니다. 다만 keypoint baseline과 fusion model까지 수행하려면 XML에서 keypoint sequence를 추출하고 정규화하는 전처리 단계가 추가로 필요합니다.

## 12. 학습 코드 구현 및 Smoke Test

4가지 실험 조건을 순서대로 학습할 수 있도록 공통 학습 코드를 구현했습니다.

구현된 주요 파일은 다음과 같습니다.

```text
src/data/dataset.py
src/data/xml_parser.py
src/data/frame_sampler.py
src/data/preprocessing.py
src/models/cnn_avg.py
src/models/cnn_lstm.py
src/models/keypoint_lstm.py
src/models/fusion.py
src/models/build.py
src/train.py
scripts/run_all_experiments.py
```

학습 대상 모델은 다음 순서로 실행됩니다.

```text
1. CNN + Average Pooling baseline
2. CNN + GRU baseline
3. 1D-CNN + GRU keypoint baseline
4. RGB + Keypoint Fusion model
```

RGB branch는 ResNet18 backbone을 사용하며, full training에서는 ImageNet pretrained weight를 사용하도록 설정했습니다. pretrained weight는 프로젝트 내부 캐시 경로에 저장했습니다.

```text
.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
```

전체 학습 실행 명령은 다음과 같습니다.

```powershell
.\.venv5070\Scripts\python.exe scripts\run_all_experiments.py
```

실험 전 전체 파이프라인이 정상 동작하는지 확인하기 위해 smoke test를 수행했습니다.

```powershell
.\.venv5070\Scripts\python.exe scripts\run_all_experiments.py --smoke
```

Smoke test 결과, 4개 모델이 모두 1 epoch 학습과 validation까지 정상적으로 완료되었습니다.

```text
baseline_cnn_avg_smoke: 정상 완료
baseline_cnn_lstm_smoke: 정상 완료
baseline_keypoint_smoke: 정상 완료
fusion_smoke: 정상 완료
```

저장 결과는 다음 경로에 생성됩니다.

```text
outputs/checkpoints/
outputs/metrics/
outputs/figures/
outputs/runs/
```

현재 상태에서는 밤새 전체 실험을 순차적으로 실행할 준비가 완료되었습니다.

추가로 full training 실행 시 `sklearn/scipy` import가 오래 걸리는 문제가 확인되어, 학습 코드에서 `sklearn` 의존성을 제거하고 accuracy, macro F1-score, confusion matrix 계산을 Numpy 기반 자체 구현으로 변경했습니다. 수정 후 smoke test를 다시 실행했으며 4개 모델 모두 정상 완료되었습니다.

## 13. 1차 학습 결과 분석

4개 실험 조건에 대한 1차 학습이 완료되었습니다. 현재 데이터셋은 AI Hub에서 `Training`과 `Validation`만 제공되었기 때문에, 본 결과는 별도 test set 결과가 아니라 **Validation split 기준 평가 결과**입니다.

현재 학습 코드는 다음 흐름으로 동작합니다.

```text
Training split으로 학습
Validation split으로 매 epoch 평가
Validation Macro F1-score가 가장 높은 checkpoint 저장
Best checkpoint를 Validation split에서 재평가
```

따라서 아래 표의 성능은 `test` 성능이 아니라 `validation` 성능입니다. 엄밀한 최종 평가를 위해서는 추후 `AI Hub Training`을 train/val로 다시 나누고, 기존 `AI Hub Validation`을 test로 사용하는 방식이 더 적절합니다.

### Validation 결과

| Experiment | Input | Model | Best Epoch | Accuracy | Macro F1 |
|---|---|---|---:|---:|---:|
| Baseline 1 | RGB frames | CNN + Average Pooling | 18 | 0.5415 | 0.5295 |
| Baseline 2 | RGB frames | CNN + GRU | 20 | 0.7138 | 0.7112 |
| Baseline 3 | Keypoints | 1D-CNN + GRU | 27 | 0.9600 | 0.9602 |
| Proposed | RGB + Keypoints | Fusion Model | 30 | 0.8938 | 0.8927 |

### Class-wise F1-score

| Experiment | 07 fall | 08 broken | 09 fire | 10 smoke | 11 abandon | 12 theft | 13 fight | 14 weak pedestrian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN + Avg | 0.677 | 0.314 | 0.651 | 0.670 | 0.296 | 0.161 | 0.467 | 1.000 |
| CNN + GRU | 0.839 | 0.422 | 0.962 | 0.852 | 0.630 | 0.286 | 0.699 | 1.000 |
| Keypoint | 1.000 | 0.917 | 0.994 | 0.906 | 0.962 | 0.929 | 0.981 | 0.994 |
| Fusion | 1.000 | 0.775 | 1.000 | 0.778 | 0.963 | 0.726 | 0.899 | 1.000 |

### 해석

1차 결과에서는 keypoint 기반 모델이 가장 높은 성능을 보였습니다. 이는 본 데이터셋의 XML keypoint가 행동 구간과 자세 변화를 직접적으로 반영하고 있기 때문으로 해석할 수 있습니다. 특히 전도, 방화, 유기, 폭행, 교통약자 클래스에서 매우 높은 F1-score를 보였습니다.

RGB 기반 모델에서는 단순 평균 pooling보다 GRU를 사용한 temporal modeling이 크게 개선되었습니다. Macro F1-score는 0.5295에서 0.7112로 증가했으며, 이는 3 FPS 저프레임 영상에서도 시간 순서 정보가 이상행동 분류에 유의미하게 작용한다는 점을 보여줍니다.

Fusion model은 RGB-only 모델보다 높은 성능을 보였지만, keypoint-only 모델보다는 낮았습니다. 현재 fusion 구조는 RGB branch와 keypoint branch를 단순 concat하는 feature-level fusion 방식이며, RGB branch backbone을 freeze한 상태입니다. 향후 backbone 일부 unfreeze, attention 기반 fusion, class imbalance 보정 등을 적용하면 성능 개선 가능성이 있습니다.

도난(theft)과 파손(broken)은 RGB 기반 모델에서 특히 어려운 클래스로 나타났습니다. 이는 객체 상호작용과 장면 맥락이 중요하지만, 현재 모델이 action segment의 제한된 16 frame만 사용하고 ResNet18 feature를 freeze한 상태이기 때문일 수 있습니다.

생성된 주요 결과 파일은 다음과 같습니다.

```text
outputs/metrics/all_experiments_full.json
outputs/metrics/all_best_validation_eval.json
outputs/metrics/*_history.csv
outputs/metrics/*_validation_best_eval.json
outputs/figures/*_validation_best_confusion.png
outputs/checkpoints/*_best.pt
```

## 14. Train/Validation/Test 분할 구조 수정

초기 1차 실험에서는 AI Hub `Training` split을 학습에 사용하고 AI Hub `Validation` split을 best checkpoint 선택과 성능 평가에 함께 사용했습니다. 이 방식에서는 validation 성능은 확인할 수 있지만, 엄밀한 의미의 test 성능이라고 보기 어렵습니다.

이를 개선하기 위해 데이터 분할 구조를 다음과 같이 수정했습니다.

```text
AI Hub Training   -> train / val 내부 분할
AI Hub Validation -> test 전용 사용
```

분할은 클래스별 비율이 유지되도록 stratified 방식으로 고정 생성했습니다. 생성 스크립트는 다음과 같습니다.

```text
scripts/build_train_val_test_manifest.py
```

생성된 split manifest는 다음 파일입니다.

```text
data/processed/frames_224_trainvaltest.csv
```

분할 결과는 다음과 같습니다.

```text
train: 4,154 samples
val:   1,037 samples
test:    650 samples
```

클래스별 test set은 기존 AI Hub Validation split을 그대로 사용합니다.

```text
07 fall: 81
08 broken: 80
09 fire: 82
10 smoke: 86
11 abandon: 80
12 theft: 80
13 fight: 80
14 weak pedestrian: 81
```

학습 코드는 이제 다음 흐름으로 동작합니다.

```text
train split으로 학습
val split으로 best checkpoint 선택
test split으로 최종 평가
```

전체 학습 및 test 평가 명령은 다음과 같습니다. 이 명령 하나로 4개 모델을 순차 학습하고, 각 모델의 best checkpoint를 test split에서 자동 평가합니다.

```powershell
.\.venv5070\Scripts\python.exe scripts\run_all_experiments.py
```

학습 완료 후 각 best checkpoint를 test set에서 별도로 다시 평가하는 명령은 다음과 같습니다.

```powershell
.\.venv5070\Scripts\python.exe scripts\evaluate_all_best.py
```

수정 후 smoke test를 통해 `train -> val best 선택 -> test 평가` 흐름이 정상 동작하는 것을 확인했습니다. 기존 1차 학습 결과와 smoke 산출물은 삭제하여, 새 구조로 full training을 다시 수행할 준비를 완료했습니다.

추가로 `scripts/run_all_experiments.py`를 수정하여, 이제 별도 평가 명령 없이도 학습이 끝난 직후 test 평가가 자동으로 수행되도록 했습니다. smoke test로 원코드 실행 흐름이 정상 동작하는 것을 확인했습니다.

## 15. 학습 곡선 저장 기능 추가

각 모델의 학습 과정을 시각적으로 확인할 수 있도록 learning curve 저장 기능을 추가했습니다. 학습이 끝나면 `src/train.py`에서 각 모델별로 다음 지표를 하나의 그림에 저장합니다.

- train/val loss
- train/val Macro F1-score
- train/val accuracy

저장 경로는 다음과 같습니다.

```text
outputs/figures/*_learning_curve.png
```

또한 이미 생성된 history CSV 파일에서 학습 곡선을 다시 생성할 수 있도록 다음 스크립트를 추가했습니다.

```text
scripts/plot_learning_curves.py
```

사용 명령은 다음과 같습니다.

```powershell
.\.venv5070\Scripts\python.exe scripts\plot_learning_curves.py
```

현재는 새 train/val/test 구조 적용 후 기존 학습 결과를 삭제한 상태이므로, full training을 다시 실행해야 learning curve가 생성됩니다.

## 16. 외부 Pretrained 모델 사용 설명

RGB 기반 모델에서는 외부 pretrained CNN backbone으로 `torchvision.models.resnet18`을 사용했습니다. ResNet18은 ImageNet 데이터셋으로 사전학습된 모델이며, 일반적인 객체, 사람, 장면, 질감 등의 시각적 특징을 추출하는 데 널리 사용됩니다.

본 프로젝트에서 ResNet18을 사용한 이유는 다음과 같습니다.

1. 현재 데이터셋은 영상 수가 5,841개로, CNN을 완전히 처음부터 학습하기에는 충분히 크지 않을 수 있습니다.
2. CCTV 영상은 3 FPS 저프레임 영상이므로, 프레임 간 세밀한 motion보다 각 프레임의 appearance feature가 중요합니다.
3. ImageNet pretrained CNN은 사람, 물체, 배경 등 기본적인 시각 특징을 이미 학습하고 있어 RGB baseline의 안정적인 feature extractor로 사용할 수 있습니다.
4. ResNet18은 ResNet 계열 중 비교적 가볍기 때문에 RTX 5070 Ti 환경에서 여러 실험을 반복하기에 적합합니다.

현재 RGB branch는 다음과 같이 사용했습니다.

```text
RGB frames
-> ResNet18 pretrained backbone
-> frame-level feature sequence
-> temporal pooling 또는 GRU
-> classifier
```

초기 실험에서는 다음 설정을 사용했습니다.

```yaml
pretrained: true
freeze_backbone: true
```

즉, ResNet18의 feature extractor는 ImageNet pretrained weight를 사용하되, backbone 파라미터는 고정했습니다. 학습되는 부분은 주로 temporal module, fusion head, classifier입니다.

이 방식은 엄밀한 의미의 full fine-tuning이라기보다는 **pretrained feature extractor 기반 학습**에 가깝습니다. 이후 성능 개선 실험에서는 다음과 같이 backbone을 일부 또는 전체 unfreeze하여 fine-tuning할 수 있습니다.

```yaml
freeze_backbone: false
```

Keypoint baseline은 RGB 이미지가 아니라 XML에서 추출한 2D 관절 좌표를 입력으로 사용하므로 외부 image pretrained model을 사용하지 않았습니다. 대신 1D-CNN과 GRU를 처음부터 학습하도록 구성했습니다.

Fusion model에서는 RGB branch에는 ResNet18 pretrained backbone을 사용하고, keypoint branch는 1D-CNN + GRU를 scratch로 학습합니다. 이후 두 branch의 feature를 concat하여 최종 classifier에 입력합니다.

```text
RGB branch: pretrained ResNet18 + GRU
Keypoint branch: 1D-CNN + GRU
Fusion: concat(video_feature, pose_feature)
Classifier: fully connected layers
```

### 모델별 구조와 특징

본 프로젝트에서는 세 가지 baseline과 하나의 proposed fusion model을 비교합니다. 각 모델은 입력 modality와 temporal modeling 방식이 다르도록 구성했습니다.

#### Baseline 1: CNN + Average Pooling

첫 번째 baseline은 RGB 프레임만 사용하는 가장 단순한 영상 분류 모델입니다.

구조는 다음과 같습니다.

```text
RGB frame sequence
-> ResNet18 pretrained CNN
-> frame-level feature vectors
-> average pooling over time
-> fully connected classifier
-> class prediction
```

이 모델에서 사용한 외부 모델은 ImageNet pretrained ResNet18입니다. ResNet18은 각 프레임에서 사람, 물체, 배경, 장면의 시각적 특징을 추출하는 feature extractor 역할을 합니다.

특징은 다음과 같습니다.

- RGB 영상만 사용합니다.
- 프레임별 시각 특징은 ResNet18로 추출합니다.
- 시간축 정보는 단순 평균 pooling으로 통합합니다.
- LSTM/GRU 같은 순서 모델이 없기 때문에 temporal order를 강하게 반영하지는 못합니다.
- 가장 단순한 RGB baseline으로, 이후 temporal model과 fusion model의 기준점 역할을 합니다.

기대 장점:

- 구조가 단순하고 학습이 안정적입니다.
- 객체, 배경, 장면 맥락을 어느 정도 반영할 수 있습니다.

한계:

- 행동의 순서나 움직임 패턴을 직접 모델링하지 못합니다.
- 전도, 폭행처럼 시간적 변화가 중요한 행동에서는 성능이 제한될 수 있습니다.

#### Baseline 2: CNN + GRU

두 번째 baseline은 RGB 프레임의 시간 순서를 반영하기 위해 CNN feature sequence에 GRU를 연결한 모델입니다.

구조는 다음과 같습니다.

```text
RGB frame sequence
-> ResNet18 pretrained CNN
-> frame-level feature sequence
-> GRU
-> final hidden state
-> fully connected classifier
-> class prediction
```

이 모델에서도 외부 pretrained 모델로 ImageNet pretrained ResNet18을 사용했습니다. ResNet18은 각 프레임의 appearance feature를 추출하고, GRU는 프레임 사이의 시간적 변화를 학습합니다.

특징은 다음과 같습니다.

- RGB 영상만 사용합니다.
- ResNet18을 통해 프레임별 feature를 추출합니다.
- GRU를 사용하여 시간 순서와 동작 변화를 반영합니다.
- CNN + Average Pooling보다 motion-dependent class에 더 적합합니다.

GRU를 사용한 이유는 다음과 같습니다.

- LSTM보다 파라미터 수가 적어 학습이 빠릅니다.
- 3 FPS 저프레임 영상에서는 복잡한 장기 motion보다 짧은 temporal transition을 잡는 것이 중요하므로 GRU가 적절합니다.
- CNN feature sequence를 입력으로 받아 영상 단위 feature를 만들 수 있습니다.

기대 장점:

- 전도, 폭행, 절도처럼 시간 흐름이 중요한 행동에서 CNN + Average Pooling보다 유리합니다.
- RGB 기반 모델 중 temporal modeling의 효과를 확인할 수 있습니다.

한계:

- RGB 정보만 사용하므로 관절 자세 변화가 명확한 행동에서는 keypoint 모델보다 불리할 수 있습니다.
- ResNet18 backbone을 freeze한 경우, CCTV 도메인에 완전히 적응하지 못할 수 있습니다.

#### Baseline 3: 1D-CNN + GRU Keypoint Model

세 번째 baseline은 XML 라벨에서 추출한 사람 keypoint 좌표만 사용하는 pose-based 모델입니다.

입력은 다음과 같습니다.

```text
keypoint sequence: [T, 34]
```

여기서 34차원은 17개 관절의 x, y 좌표를 의미합니다.

구조는 다음과 같습니다.

```text
keypoint sequence
-> 1D-CNN over temporal axis
-> GRU
-> final hidden state
-> fully connected classifier
-> class prediction
```

이 모델은 외부 pretrained image model을 사용하지 않습니다. 입력이 이미지가 아니라 2D keypoint 좌표이기 때문에, 1D-CNN과 GRU를 처음부터 학습하도록 구성했습니다.

특징은 다음과 같습니다.

- RGB 이미지를 사용하지 않고 keypoint 좌표만 사용합니다.
- x, y 좌표는 원본 영상 width/height 기준으로 0~1 범위로 정규화합니다.
- 1D-CNN은 짧은 구간의 자세 변화 패턴을 추출합니다.
- GRU는 sequence 전체의 시간적 흐름을 요약합니다.

기대 장점:

- 전도, 폭행, 교통약자처럼 사람의 자세와 움직임이 중요한 class에 강합니다.
- RGB 배경이나 조명 변화의 영향을 상대적으로 덜 받습니다.
- 입력 차원이 작아 학습 속도가 빠르고 GPU 메모리 사용량이 적습니다.

한계:

- 담배, 물건, 불꽃, 파손 물체처럼 객체 정보가 중요한 class에서는 RGB 정보가 없기 때문에 한계가 있습니다.
- XML keypoint 품질에 크게 의존합니다.
- 실제 서비스 환경에서는 keypoint 추정 모델이 추가로 필요합니다.

#### Proposed Model: RGB + Keypoint Fusion

제안 모델은 RGB 영상 feature와 keypoint motion feature를 함께 사용하는 multimodal fusion 모델입니다.

구조는 다음과 같습니다.

```text
[RGB branch]
RGB frame sequence
-> ResNet18 pretrained CNN
-> frame-level feature sequence
-> GRU
-> video feature

[Keypoint branch]
keypoint sequence
-> 1D-CNN
-> GRU
-> pose feature

[Fusion]
concat(video feature, pose feature)
-> fully connected layers
-> class prediction
```

Fusion model에서 사용한 외부 pretrained 모델은 RGB branch의 ResNet18입니다. Keypoint branch는 좌표 sequence를 입력으로 받기 때문에 scratch로 학습합니다.

특징은 다음과 같습니다.

- RGB 영상과 keypoint sequence를 모두 사용합니다.
- RGB branch는 장면, 객체, 배경, 외형 정보를 담당합니다.
- Keypoint branch는 자세, 동작, 사람 움직임 정보를 담당합니다.
- 두 branch의 feature를 concat하여 feature-level fusion을 수행합니다.

기대 장점:

- RGB-only 모델보다 사람 자세 정보를 더 잘 반영할 수 있습니다.
- Keypoint-only 모델보다 객체와 장면 맥락을 추가로 활용할 수 있습니다.
- 파손, 흡연, 절도처럼 객체/상황 정보가 필요한 class에서 keypoint-only의 한계를 보완할 수 있습니다.

한계:

- 두 modality를 동시에 사용하므로 학습 구조가 더 복잡합니다.
- 단순 concat fusion은 두 feature의 중요도를 동적으로 조절하지 못합니다.
- RGB branch를 freeze하면 CCTV 도메인에 대한 적응력이 제한될 수 있습니다.

향후 개선 방향:

- ResNet18 마지막 block 일부를 unfreeze하여 fine-tuning합니다.
- feature concat 대신 attention-based fusion을 적용합니다.
- class별 오류를 분석해 RGB와 keypoint의 contribution을 다르게 조정합니다.
- action segment sampling 전략을 개선합니다.

## 17. 224 x 224 이미지 변환 이유

원본 CCTV 영상은 1920 x 1080 해상도입니다. 모든 프레임을 원본 해상도로 학습에 사용하면 저장 용량, GPU 메모리, 학습 시간이 크게 증가합니다. 따라서 CNN 입력용 프레임을 `224 x 224` RGB JPEG로 변환했습니다.

224 x 224로 변환한 주요 이유는 다음과 같습니다.

1. ResNet18 등 ImageNet pretrained CNN은 일반적으로 224 x 224 입력 크기를 기준으로 학습되었습니다.
2. 224 x 224는 pretrained backbone과 호환성이 좋고, 별도의 구조 변경 없이 사용할 수 있습니다.
3. 원본 1920 x 1080 프레임을 그대로 사용하면 batch size를 크게 줄여야 하고 학습 시간이 길어집니다.
4. 본 실험의 목적은 고해상도 객체 검출이 아니라 이상행동 분류이므로, baseline 수준에서는 224 x 224 입력으로도 전체적인 자세, 장면, 행동 단서를 학습할 수 있습니다.
5. 모든 모델에서 동일한 입력 크기를 사용하면 baseline 간 비교가 공정해집니다.

전처리 결과는 다음과 같습니다.

```text
원본 영상 해상도: 대부분 1920 x 1080
저장 프레임 크기: 224 x 224
색상 형식: RGB
저장 형식: JPEG
저장 프레임 수: 1,052,905장
전처리 용량: 약 19.65GB
```

프레임 저장 경로는 다음과 같습니다.

```text
data/processed/frames_224/
```

각 영상은 별도의 폴더로 분리하여 저장했습니다.

```text
data/processed/frames_224/Training/07_fall/{video_stem}/frame_000000.jpg
```

이렇게 구성하면 CNN + Average Pooling, CNN + GRU, Fusion model이 동일한 frame directory와 manifest를 공유할 수 있습니다.

## 18. 최종 실험 과정 정리

현재 실험 과정은 다음 순서로 정리됩니다.

### 1. 데이터 준비

AI Hub에서 제공된 원본 zip 데이터를 프로젝트 경로로 이동한 뒤 압축을 해제했습니다. 원본 zip은 중복 저장을 피하기 위해 삭제했고, 압축 해제된 MP4/XML 파일만 유지했습니다.

```text
data/extracted/
|-- Training/
|   |-- videos/
|   `-- labels/
`-- Validation/
    |-- videos/
    `-- labels/
```

### 2. Manifest 생성

각 MP4 영상과 대응되는 XML 라벨 파일을 1:1로 매칭하여 manifest를 생성했습니다.

```text
data/splits/manifest.csv
data/processed/frames_224_manifest.csv
```

확인 결과 MP4와 XML은 각각 5,841개였고, 누락된 XML은 없었습니다.

### 3. FPS 및 프레임 수 확인

OpenCV를 사용하여 전체 영상의 FPS와 프레임 수를 확인했습니다.

```text
전체 영상 FPS: 3.0
대부분 영상 길이: 약 60초
대부분 프레임 수: 180 또는 181
```

따라서 모델 입력에서는 action segment 구간에서 일정 개수의 프레임을 uniform sampling하도록 구성했습니다.

### 4. RGB 프레임 전처리

모든 MP4 영상을 224 x 224 RGB JPEG 프레임으로 변환했습니다.

```text
data/processed/frames_224/
```

영상 단위 manifest와 프레임 단위 index를 함께 생성했습니다.

```text
data/processed/frames_224_manifest.csv
data/processed/frame_index_224.csv
```

### 5. Train/Validation/Test 분할

보고서용 실험의 엄밀성을 높이기 위해 다음 구조로 split을 재구성했습니다.

```text
AI Hub Training   -> train / val
AI Hub Validation -> test
```

분할 결과:

```text
train: 4,154 samples
val:   1,037 samples
test:    650 samples
```

최종 split manifest는 다음 파일입니다.

```text
data/processed/frames_224_trainvaltest.csv
```

### 6. 모델 학습

4개 모델을 동일한 학습 루프에서 순차적으로 학습합니다.

```text
1. CNN + Average Pooling
2. CNN + GRU
3. 1D-CNN + GRU Keypoint
4. RGB + Keypoint Fusion
```

학습 명령은 다음과 같습니다.

```powershell
.\.venv5070\Scripts\python.exe scripts\run_all_experiments.py
```

이 명령은 각 모델에 대해 다음 과정을 자동 수행합니다.

```text
train split으로 학습
val split으로 매 epoch 평가
Validation Macro F1-score 기준 best checkpoint 저장
best checkpoint를 test split에서 평가
history CSV, summary JSON, confusion matrix, learning curve 저장
```

### 7. 평가 및 결과 저장

평가 지표는 다음을 저장합니다.

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1-score
- Confusion Matrix
- Learning Curve

결과 저장 경로는 다음과 같습니다.

```text
outputs/checkpoints/
outputs/metrics/
outputs/figures/
outputs/runs/
```

학습 곡선은 다음 경로에 저장됩니다.

```text
outputs/figures/*_learning_curve.png
```

혼동행렬은 다음 경로에 저장됩니다.

```text
outputs/figures/*_confusion.png
outputs/figures/*_test_best_confusion.png
```

### 8. 실험 해석 방향

결과 분석에서는 다음 관점을 중심으로 비교합니다.

1. RGB-only 모델에서 temporal modeling이 성능을 향상시키는지 확인합니다.
2. Keypoint-only 모델이 자세 변화가 중요한 행동에서 강점을 보이는지 확인합니다.
3. Fusion model이 RGB의 장면/객체 정보와 keypoint의 자세 정보를 결합하여 성능을 개선하는지 확인합니다.
4. 클래스별 confusion matrix를 통해 파손, 절도, 흡연 등 객체 또는 상황 맥락이 중요한 클래스에서 어떤 오류가 발생하는지 분석합니다.
5. Validation best checkpoint와 test 성능 차이를 확인하여 과적합 여부를 판단합니다.

## 19. 학습 수렴 문제 확인 및 수정

train/val/test 구조로 바꾼 뒤 `baseline_cnn_avg`의 학습 곡선이 이전보다 크게 나빠지는 문제가 확인되었습니다.

문제 로그 예시는 다음과 같습니다.

```text
baseline_cnn_avg epoch 1: train_loss=2.6736 train_f1=0.1262 valid_loss=2.4652 valid_f1=0.0297
baseline_cnn_avg epoch 7: train_loss=2.4826 train_f1=0.1257 valid_loss=2.5272 valid_f1=0.0462
```

이 결과는 8-class random 수준에서 거의 벗어나지 못하는 상태였으므로, 정상적인 수렴으로 보기 어렵다고 판단했습니다.

원인 확인 결과, split 이름을 기존 `Training/Validation`에서 `train/val/test`로 바꾸면서 DataLoader의 shuffle 조건이 맞지 않았습니다. 기존 코드는 split 이름이 정확히 `Training`일 때만 shuffle을 수행했기 때문에, 새 `train` split에서는 학습 데이터가 섞이지 않았습니다. manifest는 클래스별로 정렬된 구조이므로, 모델이 한 클래스씩 몰아서 배치를 보게 되어 최적화가 불안정해질 수 있습니다.

수정 내용은 다음과 같습니다.

- `src/train.py`에서 `train`과 `Training` split 모두 shuffle되도록 수정했습니다.
- `baseline_cnn_avg`의 learning rate를 과하게 높였던 `0.001`에서 `0.0003`으로 되돌렸습니다.
- frozen ResNet18 feature를 classifier 또는 GRU에 넣기 전에 `LayerNorm`을 적용하도록 수정했습니다.
- RGB branch를 freeze한 경우 ResNet18 backbone은 train mode가 아니라 eval mode로 고정되도록 유지했습니다.

수정 후 `baseline_cnn_avg`를 3 epoch만 재검증한 결과는 다음과 같습니다.

```text
baseline_cnn_avg epoch 1: train_loss=1.6781 train_f1=0.3812 valid_loss=1.3758 valid_f1=0.5292
baseline_cnn_avg epoch 2: train_loss=1.2828 train_f1=0.5470 valid_loss=1.1632 valid_f1=0.6512
baseline_cnn_avg epoch 3: train_loss=1.1177 train_f1=0.6077 valid_loss=1.0429 valid_f1=0.6549
```

따라서 수렴 문제는 모델 구조 자체의 실패라기보다 train split shuffle 누락과 임시로 높였던 learning rate 영향이 컸다고 정리할 수 있습니다. 이후 전체 학습은 수정된 코드 기준으로 다시 수행해야 합니다.
