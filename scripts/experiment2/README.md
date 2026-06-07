# Experiment2 Scripts

2차 실험은 GT keypoint를 직접 입력하지 않고, RGB 이미지에서 keypoint를 예측한 뒤 downstream 행동 분류에 사용하는 구조입니다.

## 비교 모델

```text
1. RGB Only
2. RGB -> Predicted Keypoint Only
3. RGB + Predicted Keypoint Fusion
```

## 실행

```powershell
.\.venv5070\Scripts\python.exe scripts\experiment2\run_experiment2.py --pose-epochs 8 --classifier-epochs 15 --batch-size 32
```

빠른 동작 확인:

```powershell
.\.venv5070\Scripts\python.exe scripts\experiment2\run_experiment2.py --smoke
```

## 산출물

```text
outputs/experiment2/checkpoints/
outputs/experiment2/metrics/
outputs/experiment2/figures/
outputs/experiment2/README.md
```
