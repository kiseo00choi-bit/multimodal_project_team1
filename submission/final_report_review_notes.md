# 1차 보고서 초안 첨삭 메모

첨부한 초안은 전체 실험 방향과 핵심 결론은 실제 결과와 대체로 일치합니다. 다만 최종 repo를 재구성하면서 코드 경로와 일부 수치/표현이 바뀐 부분이 있어, 아래 항목은 제출 전 수정이 필요합니다.

## 반드시 수정할 부분

| 구분 | 초안 내용 | 실제 repo/실험 기준 수정 |
|---|---|---|
| 코드 경로 | `scripts/extract_aihub_dataset.py`, `scripts/build_manifest.py` 등 root `scripts/` 기준 | 현재는 `scripts/data_processing/extract_aihub_dataset.py`, `scripts/data_processing/build_manifest.py`처럼 `scripts/data_processing/` 아래로 이동했습니다. |
| 1차 실험 실행 코드 | 1차 모델 목록이 4개처럼 보임 | 최종 결과표에는 `RGB + GT Keypoint Cross-Attention Fusion`까지 포함하므로 5개 모델로 설명해야 합니다. `scripts/experiment1/run_experiment1.py`도 `fusion_attention.yaml`을 포함하도록 수정했습니다. |
| 2차 Pose Estimator 수치 | Valid normalized MPJPE 0.0649 | 최종 `outputs/experiment2/README.md` 기준 best valid normalized MPJPE는 `0.0648`입니다. |
| 클래스명 | `이동약자` 또는 `교통약자` 혼용 가능성 | 실제 폴더명과 `label_map.json` 기준은 `14.교통약자`, English label은 `weak_pedestrian`입니다. 보고서에서는 `교통약자`로 통일했습니다. |
| 2차 Fusion 모델명 | `RGB + RGB -> Predicted Keypoint Fusion`처럼 보이는 표현 | 보고서와 발표에서는 `RGB + Predicted Keypoint Fusion`으로 쓰는 것이 명확합니다. RGB branch와 RGB에서 예측한 keypoint branch를 결합한다는 설명은 본문에서 풀어 쓰면 됩니다. |
| 데이터 수 설명 | AI Hub 구축 실적 약 6,400건과 실제 사용 clip 수 혼동 가능 | 실제 실험에는 MP4/XML이 1:1로 매칭된 `5,841개 clip`을 사용했습니다. AI Hub의 구축 목표/실적 수와 로컬에서 사용한 매칭 clip 수는 구분해서 설명해야 합니다. |
| 제출 코드 설명 | 예전 구조 기준 설명 | 최종 제출 코드는 `src/`, `scripts/data_processing/`, `scripts/experiment1/`, `scripts/experiment2/`, `scripts/demo/` 기준으로 설명해야 합니다. |

## 실제 실험 기준 핵심 정리

### 데이터

- 실제 사용 clip 수: 5,841개
- split: Train 4,154 / Validation 1,037 / Test 650
- Test는 AI Hub `Validation` split 전체를 사용했습니다.
- RGB frame은 ResNet18 입력 호환성과 메모리 절약을 위해 224x224로 변환했습니다.
- keypoint 좌표는 원본 해상도 기준 pixel 좌표를 0~1 범위로 정규화했습니다.

### 1차 실험

1차 실험은 XML GT keypoint를 직접 입력에 사용한 성능 상한선 확인 실험입니다.

| 모델 | Test Macro F1 |
|---|---:|
| CNN + Average Pooling | 0.6246 |
| CNN + GRU | 0.7547 |
| GT Keypoint 1D-CNN + GRU | 0.9497 |
| RGB + GT Keypoint Fusion | 0.8467 |
| RGB + GT Keypoint Cross-Attention Fusion | 0.8620 |

해석은 “GT keypoint가 주어지면 자세 정보가 매우 강력하지만, 실제 CCTV 추론에서는 GT keypoint를 사용할 수 없으므로 1차 결과는 상한선에 가깝다”로 쓰는 것이 정확합니다.

### 2차 실험

2차 실험은 실제 추론 환경에 가깝게 RGB 이미지에서 keypoint를 예측한 뒤 downstream 분류에 사용하는 실험입니다.

| 모델 | Test Macro F1 |
|---|---:|
| RGB Only | 0.7228 |
| RGB -> Predicted Keypoint Only | 0.5980 |
| RGB + Predicted Keypoint Fusion | 0.7270 |

해석은 “Predicted keypoint는 단독으로는 RGB보다 낮지만, RGB와 결합하면 test Macro F1이 0.7228에서 0.7270으로 소폭 상승했다”가 가장 안전합니다.

## 새 repo 기준 코드 설명

보고서 코드 설명은 다음 구조를 기준으로 작성하는 것이 맞습니다.

| 경로 | 설명 |
|---|---|
| `src/data/` | Dataset, XML parser, frame sampler, preprocessing |
| `src/models/` | 1차 실험용 RGB, keypoint, fusion, cross-attention 모델 |
| `src/train.py` | 1차 실험 공통 train/validation loop, best checkpoint 저장 |
| `src/evaluate.py` | best checkpoint test 평가 |
| `scripts/data_processing/` | AI Hub 압축 해제, manifest 생성, FPS 확인, frame 추출, split 생성 |
| `scripts/experiment1/` | 1차 실험 실행 코드와 config |
| `scripts/experiment2/run_experiment2.py` | Pose Estimator와 2차 downstream 모델을 한 번에 실행하는 코드 |
| `scripts/demo/make_realtime_cctv_demo.py` | 실제 CCTV clip 기반 시연 영상 생성 및 inference time 측정 |
| `outputs/experiment1/` | 1차 실험 고정 결과 archive |
| `outputs/experiment2/` | 2차 실험 최종 결과 |
| `submission/experiment1/` | 1차 제출용 요약과 결과표 |
| `submission/experiment2/` | 2차 제출용 요약과 결과표 |

초안의 `scripts/extract_frames_224.py`처럼 root `scripts/`에 있는 것으로 적은 부분은 모두 `scripts/data_processing/extract_frames_224.py`처럼 새 구조에 맞게 수정해야 합니다.

## 반영 완료한 수정

- `submission/final_report_draft.md`에서 `이동약자`를 `교통약자`로 통일했습니다.
- `submission/final_report_draft.md`의 코드 설명 config 목록에 `fusion_attention.yaml`을 추가했습니다.
- `scripts/experiment1/run_experiment1.py` 기본 CONFIGS에 `fusion_attention.yaml`을 추가했습니다.
- `scripts/experiment1/README.md`의 비교 모델 목록에 cross-attention fusion을 추가했습니다.

