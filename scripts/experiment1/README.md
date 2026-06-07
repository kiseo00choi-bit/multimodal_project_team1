# Experiment1 Scripts

1차 실험은 XML에 라벨링된 GT keypoint를 직접 사용하는 baseline 비교입니다.

보고서용 고정 결과는 재학습하지 않고 `outputs/experiment1/`에 기존 non-smoke 결과물을 archive해 두었습니다. 아래 실행 명령은 재현 또는 재학습이 필요할 때만 사용합니다.

## 비교 모델

```text
1. RGB CNN + Average Pooling
2. RGB CNN + GRU
3. GT Keypoint 1D-CNN + GRU
4. RGB + GT Keypoint Fusion
5. RGB + GT Keypoint Cross-Attention Fusion
```

## 실행

```powershell
.\.venv5070\Scripts\python.exe scripts\experiment1\run_experiment1.py --batch-size 32
```

빠른 동작 확인:

```powershell
.\.venv5070\Scripts\python.exe scripts\experiment1\run_experiment1.py --smoke
```

## 산출물

```text
scripts/experiment1/configs/       # archived training configs
outputs/experiment1/runs/
outputs/experiment1/checkpoints/
outputs/experiment1/metrics/
outputs/experiment1/figures/
```

현재 `outputs/experiment1/README.md`에는 기존 1차 실험의 고정 결과표가 정리되어 있습니다.
