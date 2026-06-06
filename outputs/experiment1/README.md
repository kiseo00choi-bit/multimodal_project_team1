# Experiment1 Archived Results

이 폴더는 기존 1차 실험 결과를 재학습 없이 정리한 archive입니다. 최종 보고서에서 1차 실험 숫자가 바뀌면 안 되므로, 루트 `outputs/`에 남아 있던 non-smoke 결과물을 그대로 복사했습니다.

## 고정 결과

| Model | Best Epoch | Valid Macro F1 | Test Accuracy | Test Macro F1 |
|---|---:|---:|---:|---:|
| CNN + Average Pooling | 19 | 0.8263 | 0.6323 | 0.6246 |
| CNN + GRU | 17 | 0.9611 | 0.7554 | 0.7547 |
| GT Keypoint 1D-CNN + GRU | 30 | 0.9864 | 0.9492 | 0.9497 |
| RGB + GT Keypoint Fusion | 21 | 0.9805 | 0.8523 | 0.8467 |
| RGB + GT Keypoint Cross-Attention Fusion | 27 | 0.9826 | 0.8646 | 0.8620 |

## 사용 기준

- 보고서/발표용 1차 실험 결과는 이 폴더의 파일을 기준으로 사용합니다.
- 숫자를 유지해야 하므로 최종 정리 단계에서는 1차 실험을 재학습하지 않습니다.
- `fusion_attention`은 기본 1차 실험 4종 이후 추가한 개선 실험입니다.

## 주요 파일

```text
metrics/all_experiments_full.json
metrics/*_history.csv
metrics/*_summary.json
metrics/*_test_best_eval.json
figures/*_confusion.png
figures/*_test_best_confusion.png
figures/learning_curves/*.png
checkpoints/*_best.pt
runs/*/config.json
```
