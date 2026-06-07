# 제출 패키지 안내

이 폴더는 최종 제출물을 준비하기 위한 정리 폴더입니다.

## 제출 항목

| 제출 항목 | 준비 파일 | 비고 |
|---|---|---|
| 1차 보고서 | `submission/final_report_draft.md` | Word 파일로 변환/편집 후 제출 |
| 발표 자료 | 별도 PPT 작성 | README/보고서의 그림과 표 사용 |
| 코드 | `submission/team1_code_submission.zip` | `.py` 파일이 5개 이상이므로 코드만 압축 |
| 시연 동영상 | 별도 녹화 | 30초~1분 |

## 코드 압축 대상

코드 zip에는 다음만 포함합니다.

```text
src/
scripts/data_processing/
scripts/experiment1/
scripts/experiment2/
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

## 시연 영상 구성 추천

30초~1분 시연은 다음 순서가 적합합니다.

1. GitHub/프로젝트 폴더 구조 표시
2. README의 데이터 가공 및 모델 구조 그림 표시
3. 2차 실험 결과표 이미지 표시
4. smoke test 명령 또는 저장된 결과 폴더 표시
