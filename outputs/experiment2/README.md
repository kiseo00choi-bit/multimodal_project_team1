# Experiment2 후속 실험 결과

## 실험 목적

라벨링된 XML keypoint를 GT로 사용해 RGB 이미지만으로 keypoint를 예측하는 모델을 학습한 뒤,
RGB-only, 예측 keypoint-only, RGB + 예측 keypoint 멀티모달 분류기를 비교했다.

## 설정

- 실행 모드: full
- manifest: `data/processed/frames_224_trainvaltest.csv`
- 입력 프레임 수: 16
- image size: 224
- pose estimator epochs: 8
- classifier epochs: 15
- batch size: 32

## Image -> Keypoint 예측기

- best epoch: 8
- best valid normalized MPJPE: 0.0648
- checkpoint: `outputs\experiment2\checkpoints\image_keypoint_estimator_best.pt`

## Downstream 분류 비교

| Model | Best Epoch | Best Valid Macro F1 | Test Accuracy | Test Macro F1 |
|---|---:|---:|---:|---:|
| RGB Only | 13 | 0.9264 | 0.7231 | 0.7228 |
| RGB -> Predicted Keypoint Only | 13 | 0.7605 | 0.5908 | 0.5980 |
| RGB + RGB -> Predicted Keypoint Fusion | 15 | 0.9450 | 0.7246 | 0.7270 |

## 해석

- Fusion의 Test Macro F1 변화량 vs RGB-only: +0.0042
- Fusion의 Test Macro F1 변화량 vs predicted keypoint-only: +0.1289
- 이 실험은 GT keypoint를 직접 입력한 기존 baseline보다 실제 추론 환경에 가깝다.
- 성능이 낮게 나오더라도 이는 keypoint 예측 오차가 downstream 분류에 누적되기 때문이다.
- RGB-only보다 predicted keypoint-only가 높으면 자세 정보 추정이 행동 분류에 유효하다고 해석한다.
- Fusion이 두 단일 입력 모델보다 높으면 RGB 정보와 예측 keypoint가 서로 보완한다고 해석한다.
- Fusion이 낮거나 비슷하면 pose estimator 품질, fusion 구조, end-to-end fine-tuning 개선이 필요하다고 해석한다.

## 산출물

- `metrics/image_keypoint_estimator_history.csv`
- `metrics/rgb_only_history.csv`
- `metrics/pred_keypoint_only_history.csv`
- `metrics/pred_keypoint_fusion_history.csv`
- `metrics/experiment2_summary.json`
- `figures/*learning_curve.png`
- `figures/downstream_learning_curves_shared_axes.png`
- `figures/*confusion.png`
- `figures/downstream_comparison.png`
