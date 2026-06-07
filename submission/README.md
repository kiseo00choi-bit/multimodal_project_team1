# 제출 패키지 안내

이 폴더는 최종 제출물을 준비하기 위한 정리 폴더입니다.

## 제출 항목

| 제출 항목 | 준비 파일 | 비고 |
|---|---|---|
| 1차 보고서 | `submission/final_report_draft.md` | Word 파일로 변환/편집 후 제출 |
| 발표 자료 | 별도 PPT 작성 | README/보고서의 그림과 표 사용 |
| 코드 | `submission/team1_code_submission.zip` | `.py` 파일이 5개 이상이므로 코드만 압축 |
| 시연 동영상 | `submission/cctv_realtime_demo.mp4` | 실제 CCTV clip 기반 데모 |
| 추론 시간 로그 | `submission/cctv_realtime_demo_timing.json` | RTX 5070 Ti 기준 측정 |

## 코드 압축 대상

코드 zip에는 다음만 포함합니다.

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

포함하지 않는 항목:

```text
data/
outputs/
.venv5070/
__pycache__/
*.pt checkpoint
```

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
.\.venv5070\Scripts\python.exe scripts\demo\make_realtime_cctv_demo.py --class-id 0 --sample-index 7 --output outputs\demo\cctv_realtime_demo.mp4 --max-seconds 60 --display-lead-frames 8
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

영상에 표시되는 skeleton은 XML GT가 아니라 `ImageKeypointEstimator`가 RGB frame만 보고 예측한 predicted keypoint입니다. 따라서 사람 detector 없이 전체 CCTV frame에서 관절 좌표를 직접 회귀하는 현재 구조에서는 사람이 작거나 배경이 복잡한 구간에서 skeleton이 사람 위치와 어긋날 수 있습니다. 이전 버전에서는 fall alert 이전 구간의 keypoint 표시를 숨겼지만, 현재 버전은 실제 실시간 pose 추론처럼 모든 frame에 predicted keypoint를 표시합니다.

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
Frame-wise pose model-only: 0.28 ms/frame
Frame-wise pose end-to-end: 9.18 ms/frame, 약 108.88 FPS
16-frame fusion clip inference: 181.95 ms/clip
16-frame equivalent throughput: 약 87.94 frame/s
```

원본 CCTV가 약 3 FPS 데이터였기 때문에 단일 CCTV stream 기준으로는 실시간 적용 가능성이 있습니다. 다만 여러 카메라를 동시에 처리하려면 batching, 영상 decode 최적화, 사람 검출기를 결합한 pose estimation 개선이 필요합니다.
