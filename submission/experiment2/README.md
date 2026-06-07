# 2차 실험 제출 요약

## 목적

2차 실험은 실제 CCTV 추론 환경에 더 가깝게 설계한 실험입니다. 1차 실험처럼 XML GT keypoint를 분류기에 직접 넣지 않고, RGB 이미지에서 keypoint를 예측하는 pose estimator를 먼저 학습한 뒤 predicted keypoint를 downstream 행동 분류에 사용했습니다.

## 실험 흐름

```text
Stage 1: RGB image -> predicted keypoint estimator 학습
Stage 2: RGB only / predicted keypoint only / RGB + predicted keypoint fusion 비교
```

## 사용 데이터

| 항목 | 내용 |
|---|---|
| 입력 RGB | 224x224 action segment frame sequence |
| pose estimator GT | XML GT keypoint |
| downstream keypoint | pose estimator가 예측한 predicted keypoint |
| split | AI Hub Training -> train/validation, AI Hub Validation -> test |
| test sample 수 | 650개 |

## 비교 모델

| 모델 | 입력 | 목적 |
|---|---|---|
| RGB Only | RGB | 2차 실험 기준 baseline |
| RGB -> Predicted Keypoint Only | predicted keypoint | 예측 자세 정보만의 성능 확인 |
| RGB + Predicted Keypoint Fusion | RGB + predicted keypoint | RGB와 예측 자세 정보 결합 효과 확인 |

## 주요 결과

Pose estimator best validation normalized MPJPE는 0.0648입니다.

| 모델 | Test Macro F1 |
|---|---:|
| RGB Only | 0.7228 |
| RGB -> Predicted Keypoint Only | 0.5980 |
| RGB + Predicted Keypoint Fusion | 0.7270 |

결과표 이미지는 `submission/experiment2/experiment2_results_table.png`에 포함했습니다.

## 시연 영상과 실시간성

제출용 시연 영상 `submission/cctv_realtime_demo.mp4`는 2차 실험의 `RGB + Predicted Keypoint Fusion` 모델 결과입니다. 영상의 skeleton은 XML GT가 아니라 RGB frame에서 예측한 predicted keypoint이며, 빈 공간에 불안정한 raw pose가 뜨지 않도록 `pose_display_mode=action`으로 행동 구간에서만 표시했습니다.

RTX 5070 Ti 기준 추론 시간:

| 항목 | 측정값 |
|---|---:|
| Frame-wise pose end-to-end | 9.24 ms/frame |
| 16-frame fusion clip inference | 179.19 ms/clip |
| 16-frame equivalent throughput | 89.29 frame/s |

## 관련 코드와 결과 위치

```text
scripts/experiment2/
scripts/demo/make_realtime_cctv_demo.py
outputs/experiment2/
outputs/demo/
docs/assets/result_tables/experiment2_results_table.png
```

