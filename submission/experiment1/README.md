# 1차 실험 제출 요약

## 목적

1차 실험은 XML에 라벨링된 GT keypoint를 직접 사용할 때 RGB 정보와 자세 정보가 이상행동 분류에 얼마나 기여하는지 비교한 baseline 실험입니다. 즉 실제 배포 모델이라기보다, GT pose 정보가 충분히 정확할 때 얻을 수 있는 상한 성능을 확인하는 단계입니다.

## 사용 데이터

| 항목 | 내용 |
|---|---|
| 입력 RGB | 224x224로 변환한 action segment frame sequence |
| 입력 keypoint | XML GT keypoint, 17개 관절 x/y 좌표 |
| split | AI Hub Training -> train/validation, AI Hub Validation -> test |
| test sample 수 | 650개 |

## 비교 모델

| 모델 | 입력 | 설명 |
|---|---|---|
| CNN + Average Pooling | RGB | ResNet18 frame feature 평균 |
| CNN + GRU | RGB | ResNet18 frame feature sequence를 GRU로 분류 |
| GT Keypoint 1D-CNN + GRU | XML GT keypoint | 관절 좌표 sequence 기반 분류 |
| RGB + GT Keypoint Fusion | RGB + XML GT keypoint | RGB branch와 pose branch concat |
| RGB + GT Keypoint Cross-Attention | RGB + XML GT keypoint | pose feature가 RGB feature를 attention으로 참조 |

## 주요 결과

| 모델 | Test Macro F1 |
|---|---:|
| CNN + Average Pooling | 0.6246 |
| CNN + GRU | 0.7547 |
| GT Keypoint 1D-CNN + GRU | 0.9497 |
| RGB + GT Keypoint Fusion | 0.8467 |
| RGB + GT Keypoint Cross-Attention | 0.8620 |

결과표 이미지는 `submission/experiment1/experiment1_results_table.png`에 포함했습니다.

## 관련 코드와 결과 위치

```text
scripts/experiment1/
outputs/experiment1/
docs/assets/result_tables/experiment1_results_table.png
```

