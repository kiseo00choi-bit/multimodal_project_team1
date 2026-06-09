# 제출 패키지 안내

이 폴더는 최종 제출물을 준비하기 위한 정리 폴더입니다.

## 제출 항목

| 제출 항목 | 준비 파일 | 비고 |
|---|---|---|
| 1차 보고서 | `submission/final_report_draft_formal.docx` | 습니다체로 정리한 Word 제출용 초안 |
| 보고서 원본 | `submission/final_report_draft.md` | 수정/재변환용 Markdown 원본 |
| 첨삭 메모 | `submission/final_report_review_notes.md` | 첨부 초안과 실제 실험/repo 차이 정리 |
| 발표 자료 | 별도 PPT 작성 | README/보고서의 그림과 표 사용 |
| 코드 | `submission/team1_code_submission.zip` | `.py` 파일이 5개 이상이므로 코드만 압축 |
| 시연 동영상 | `submission/cctv_realtime_demo.mp4` | 실제 CCTV clip 기반 데모 |
| 추론 시간 로그 | `submission/cctv_realtime_demo_timing.json` | RTX 5070 Ti 기준 측정 |

## 1차 실험 / 2차 실험 파일 구분

| 구분 | 목적 | 제출 폴더/파일 | 주요 내용 |
|---|---|---|---|
| 데이터 가공 공통 | MP4/XML 정리, frame 추출, split 생성 | `scripts/data_processing/` | 1차와 2차 실험이 공통으로 사용하는 데이터 전처리 코드 |
| 1차 실험 | RGB baseline과 XML GT keypoint 기반 모델 비교 | `submission/experiment1/` | GT keypoint를 직접 사용한 baseline 결과 요약과 결과표 |
| 1차 실험 코드 | 1차 모델 학습/평가 | `scripts/experiment1/` | RGB CNN, RGB GRU, GT keypoint, RGB+GT keypoint fusion |
| 1차 실험 원본 결과 | 1차 결과 archive | `outputs/experiment1/` | metrics, figures, run config, checkpoint 저장 위치 |
| 2차 실험 | 실제 추론 환경을 반영한 predicted keypoint 기반 비교 | `submission/experiment2/` | RGB-only, predicted keypoint-only, RGB+predicted keypoint fusion 결과 요약과 결과표 |
| 2차 실험 코드 | pose estimator와 downstream classifier 학습/평가 | `scripts/experiment2/` | RGB image -> keypoint 예측 후 3개 downstream 모델 비교 |
| 2차 실험 원본 결과 | 2차 결과 archive | `outputs/experiment2/` | metrics, learning curve, train log, checkpoint 저장 위치 |
| 2차 시연 | 실제 CCTV clip 추론 | `submission/cctv_realtime_demo.mp4` | 2차 실험의 RGB + Predicted Keypoint Fusion 모델 사용 |

요약하면 1차 실험은 `GT keypoint를 직접 넣었을 때의 성능 확인`, 2차 실험은 `실제처럼 RGB에서 keypoint를 예측한 뒤 사용하는 구조 확인`입니다.

## 코드 압축 대상

코드 zip에는 학습 코드, 데이터 가공 코드, 파라미터 파일만 포함합니다.

```text
src/data/                     # Dataset, XML parser, frame sampling, preprocessing
src/models/                   # 1차/2차 RGB, keypoint, fusion model definitions
src/train.py, src/evaluate.py # common training, validation, test evaluation utilities
scripts/data_processing/      # AI Hub unzip, manifest, FPS check, frame extraction, split scripts
scripts/experiment1/          # experiment1 runner: GT keypoint baseline and fusion experiments
scripts/experiment2/          # experiment2 runner: RGB -> predicted keypoint -> classification
configs/                      # experiment1 YAML configs and experiment2 default parameter JSON
requirements.txt              # Python package list
```

포함하지 않는 항목:

```text
data/
outputs/
.venv5070/
__pycache__/
*.pt checkpoint
scripts/demo/
scripts/plot_*.py
README.md
```

2차 실험은 별도 YAML config를 사용하지 않고 `scripts/experiment2/run_experiment2.py`의 argparse 인자로 설정을 관리합니다. 제출 ZIP에는 기본 실행값을 확인할 수 있도록 `configs/experiment2_default_params.json`을 함께 포함했습니다. 2차 모델 정의는 실행 파일에 직접 두지 않고 `src/models/experiment2.py`로 분리했습니다.

## 재현 명령

가상환경 활성화:

```powershell
.\.venv5070\Scripts\Activate.ps1
```

2차 실험 smoke test:

```powershell
.\.venv5070\Scripts\python.exe scripts\experiment2\run_experiment2.py --smoke --output-dir outputs\submission_smoke
```

최근 smoke test 확인 결과:

```text
device: cuda
dataset sizes: train=16, val=16, test=16
pose estimator: 1 epoch 완료
rgb_only classifier: 1 epoch 완료
pred_keypoint_only classifier: 1 epoch 완료
pred_keypoint_fusion classifier: 1 epoch 완료
summary: outputs/submission_smoke/README.md 생성 확인
```

2차 실험 full run:

```powershell
.\.venv5070\Scripts\python.exe scripts\experiment2\run_experiment2.py --pose-epochs 8 --classifier-epochs 15 --batch-size 32
```

결과표 이미지 재생성:

```powershell
.\.venv5070\Scripts\python.exe scripts\plot_result_tables.py
```

모델 구조도 재생성:

```powershell
.\.venv5070\Scripts\python.exe scripts\plot_model_architecture_diagrams.py
```

실제 CCTV clip 기반 시연 영상 생성:

```powershell
.\.venv5070\Scripts\python.exe scripts\demo\make_realtime_cctv_demo.py --class-id 0 --sample-index 7 --output outputs\demo\cctv_realtime_demo.mp4 --max-seconds 60 --display-lead-frames 8 --pose-display-mode action
```

## 시연 영상 설명

`submission/cctv_realtime_demo.mp4`는 AI Hub test split의 실제 CCTV clip을 사용하여 생성했습니다. 2차 실험의 `RGB + Predicted Keypoint Fusion` best checkpoint로 clip을 예측하고, CCTV 화면 위에 다음 정보를 overlay했습니다.

```text
LIVE CCTV MONITOR
Prediction label
Confidence
GT label
Top-3 probabilities
Predicted keypoint skeleton
Frame progress
Action segment 표시
```

영상에 표시되는 skeleton은 XML GT가 아니라 `ImageKeypointEstimator`가 RGB frame만 보고 예측한 predicted keypoint입니다. 따라서 사람 detector 없이 전체 CCTV frame에서 관절 좌표를 직접 회귀하는 현재 구조에서는 사람이 작거나 배경이 복잡한 구간에서 skeleton이 사람 위치와 어긋날 수 있습니다. 제출용 데모는 빈 공간에 불안정한 raw pose가 표시되지 않도록 `--pose-display-mode action`을 사용하여 fall alert 직전/행동 구간에서만 predicted keypoint를 표시합니다. 연구용으로 raw pose 결과를 모든 frame에서 확인하려면 `--pose-display-mode always`를 사용할 수 있습니다.

생성된 예시 영상의 결과:

```text
GT label: fall
Prediction: fall
Confidence: 0.7929
Fall alert starts at frame: 110
Frames written: 180
Display FPS: 6.00
Resolution: 1280x720
```

RTX 5070 Ti 환경에서 측정한 추론 시간:

```text
Frame-wise pose model-only: 0.27 ms/frame
Frame-wise pose end-to-end: 9.24 ms/frame, 약 108.19 FPS
16-frame fusion clip inference: 179.19 ms/clip
16-frame equivalent throughput: 약 89.29 frame/s
```

원본 CCTV가 약 3 FPS 데이터였기 때문에 단일 CCTV stream 기준으로는 실시간 적용 가능성이 있습니다. 다만 여러 카메라를 동시에 처리하려면 batching, 영상 decode 최적화, 사람 검출기를 결합한 pose estimation 개선이 필요합니다.
