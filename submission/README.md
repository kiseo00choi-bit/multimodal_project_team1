# 제출 패키지 안내

이 폴더는 최종 제출물을 준비하기 위한 정리 폴더입니다.

## 제출 항목

| 제출 항목 | 준비 파일 | 비고 |
|---|---|---|
| 1차 보고서 | `submission/final_report_draft.md` | Word 파일로 변환/편집 후 제출 |
| 발표 자료 | 별도 PPT 작성 | README/보고서의 그림과 표 사용 |
| 코드 | `submission/team1_code_submission.zip` | `.py` 파일이 5개 이상이므로 코드만 압축 |
| 시연 동영상 | `submission/cctv_realtime_demo.mp4` | 약 30초 |

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
.\.venv5070\Scripts\python.exe scripts\demo\make_realtime_cctv_demo.py --class-id 0 --sample-index 0 --output outputs\demo\cctv_realtime_demo.mp4
```

## 시연 영상 설명

`submission/cctv_realtime_demo.mp4`는 AI Hub test split의 실제 CCTV clip을 사용하여 생성했습니다. 2차 실험의 `RGB + Predicted Keypoint Fusion` best checkpoint로 clip을 예측하고, CCTV 화면 위에 다음 정보를 overlay했습니다.

```text
LIVE CCTV MONITOR
Prediction label
Confidence
GT label
Top-3 probabilities
Frame progress
Action segment 표시
```

생성된 예시 영상의 결과:

```text
GT label: fall
Prediction: fall
Confidence: 0.7929
Fall alert starts at frame: 110
Duration: 약 30초
```
