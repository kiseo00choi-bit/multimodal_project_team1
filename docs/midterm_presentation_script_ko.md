# 중간발표 7분 발표 대본

## 발표 전 수정/주의 메모

- 6번 슬라이드의 `CNN + LSTM`은 실제 실험 기준으로 `CNN + GRU`라고 설명한다.
- `KeyPoint Only (1D-CNN)`은 실제로는 `1D-CNN + GRU Keypoint` 구조라고 설명한다.
- Keypoint는 224x224로 바꾼 것이 아니라, 원본 XML 좌표를 원본 해상도 기준으로 0~1 정규화해서 사용했다.
- `6,492 clips`는 AI Hub 구축량, `5,841 clips`는 실제 로컬에서 영상/XML 매칭 후 사용한 수량이다.

## 1. 제목

안녕하세요. 저희는 CCTV 영상과 사람 포즈 정보를 결합한 멀티모달 이상행동 분류 모델을 주제로 프로젝트를 진행했습니다.

핵심 목표는 RGB 영상만 사용했을 때와, 사람의 관절 좌표인 Keypoint 정보를 함께 사용했을 때 이상행동 분류 성능이 어떻게 달라지는지 비교하는 것입니다.

## 2. 프로젝트 주제 및 동기

무인 점포가 늘어나면서 CCTV 관제의 중요성이 커지고 있습니다. 하지만 CCTV는 사람이 계속 영상을 확인해야 한다는 한계가 있습니다.

그래서 저희는 CCTV 영상에서 전도, 절도, 폭행, 흡연 같은 이상행동을 자동으로 분류하는 모델을 만들고자 했습니다.

RGB 영상은 매장 배경과 외형 정보를 담고, Keypoint는 사람의 자세와 움직임 구조를 담습니다. 두 정보를 함께 쓰면 단일 영상 모델보다 더 안정적인 행동 인식이 가능할 것이라고 보았습니다.

## 3. Dataset

사용한 데이터는 AI Hub의 실내 편의점, 매장 사람 이상행동 데이터입니다.

AI Hub 설명에는 전체 구축량이 6,492 clips로 되어 있지만, 실제로 다운로드 후 영상과 XML 라벨이 매칭되는 기준으로 확인한 데이터는 5,841 clips였습니다.

데이터는 MP4 영상과 XML annotation으로 구성되어 있고, 클래스는 전도, 파손, 방화, 흡연, 유기, 절도, 폭행, 이동약자 총 8개입니다. 전체 프레임은 약 105만 장입니다.

## 4. Dataset 가공

원본 영상은 대부분 1920x1080 해상도이고, 약 3fps로 구성되어 있습니다.

RGB 프레임은 ResNet18 입력에 맞추기 위해 224x224 이미지로 변환했습니다. 224x224는 ImageNet 사전학습 모델에서 일반적으로 쓰는 입력 크기라 전이학습에 적합합니다.

XML에서는 행동 시작/종료 프레임, 사람 bbox, 17개 관절 Keypoint를 추출했습니다. Keypoint 좌표는 원본 해상도 기준 픽셀 좌표이기 때문에 width와 height로 나누어 0에서 1 사이 값으로 정규화했습니다.

데이터 분할은 AI Hub Training 5,191개를 Train 4,154개와 Validation 1,037개로 나누고, AI Hub Validation 650개를 최종 Test set으로 사용했습니다.

## 5. 관련 연구

관련 연구는 세 방향을 참고했습니다.

첫 번째는 pose 기반 이상행동 탐지입니다. 자세 정보는 배경이나 조명 변화의 영향을 덜 받기 때문에 행동 인식에 유용합니다.

두 번째는 RGB Frame과 Skeleton Sequence를 결합하는 Fusion 방식입니다. RGB는 장면 정보를, Skeleton은 사람의 움직임 구조를 담기 때문에 서로 보완적입니다.

세 번째는 ST-GCN 같은 Skeleton 기반 행동 인식입니다. 관절을 graph로 보고 시간 흐름까지 학습하는 방식인데, 이번 프로젝트에서는 구현 범위를 고려해 1D-CNN과 GRU 기반 구조부터 적용했습니다.

## 6. 실험 목표

실험 질문은 RGB 영상 기반 단일모달보다 Keypoint를 활용한 멀티모달이 더 효과적인가입니다.

비교한 모델은 네 계열입니다. 첫 번째는 CNN feature를 평균내는 RGB Average Pooling 모델입니다. 두 번째는 RGB feature의 시간 흐름을 보는 CNN-GRU 모델입니다.

세 번째는 XML Keypoint만 사용하는 Keypoint-only 모델입니다. 네 번째는 RGB feature와 Keypoint feature를 함께 사용하는 Fusion 모델입니다.

평가지표는 Accuracy, Macro F1-score, Confusion Matrix를 사용했고, 클래스별 성능을 균형 있게 보기 위해 Macro F1을 주요 지표로 보았습니다.

## 7. 예상 리스크

리스크는 크게 네 가지였습니다.

첫째, 전체 프레임 수가 많아 학습 비용이 크기 때문에 행동 구간에서 16프레임을 샘플링했습니다.

둘째, 데이터 누수를 막기 위해 프레임 단위가 아니라 영상 단위로 Train, Validation, Test를 나누었습니다.

셋째, 학습 자원을 고려해 RGB 모델은 224x224 resize와 ResNet18 기반 feature extractor를 사용했습니다.

마지막으로, 실제 CCTV에서는 GT Keypoint가 주어지지 않기 때문에 현재 Keypoint 실험은 실제 적용 전 단계의 성능 확인이라는 한계가 있습니다.

## 8. 결과

결과를 보면 CNN Average Pooling은 Test Macro F1이 0.6246으로 가장 낮았습니다. 단순 평균 방식이라 시간적 행동 변화가 충분히 반영되지 않은 것으로 보입니다.

CNN-GRU는 Test Macro F1이 0.7547로 올라갔습니다. RGB만 사용해도 시간 흐름을 반영하면 성능이 개선된다는 것을 확인했습니다.

Keypoint-only 모델은 Test Macro F1 0.9497로 가장 높았습니다. 다만 이 모델은 XML의 정답 Keypoint를 직접 사용하기 때문에, 실제 적용 모델이라기보다는 관절 정보가 얼마나 강력한지 보여주는 상한선에 가깝습니다.

Fusion 모델은 0.8467, Cross-Attention Fusion은 0.8620을 기록했습니다. 단순 결합보다 Attention을 통해 RGB와 Keypoint가 상호작용하도록 만든 구조가 더 좋은 성능을 보였습니다.

## 9. Learning Curve

Learning Curve를 보면 Average Pooling은 성능이 낮고 천천히 수렴합니다. 반면 GRU와 Fusion 계열은 초반부터 더 높은 Macro F1에 도달합니다.

Keypoint-only는 가장 높은 성능을 보였지만, loss curve는 불안정합니다. 이유는 입력이 이미지 전체가 아니라 관절 좌표 몇 개이기 때문에 일부 좌표 누락이나 튐에 민감하기 때문입니다.

또한 일부 샘플을 높은 확신으로 틀리면 F1은 크게 떨어지지 않아도 CrossEntropy Loss는 크게 증가할 수 있습니다. 그래서 Keypoint-only는 성능은 높지만, 실제 환경에서는 pose estimation noise까지 고려해야 합니다.

## 10. Further Work

현재 실험에서는 Keypoint-only가 가장 높은 성능을 냈지만, 실제 CCTV 환경에서는 GT Keypoint가 바로 제공되지 않습니다.

그래서 다음 단계는 XML Keypoint를 정답으로 사용해 이미지에서 Keypoint를 예측하는 모델을 학습하는 것입니다.

그 후 예측된 Keypoint만 사용하는 모델과, 예측 Keypoint와 RGB 이미지를 함께 사용하는 멀티모달 모델을 비교할 수 있습니다.

이렇게 하면 현재의 정답 Keypoint 기반 실험을 실제 추론 환경에 가까운 모델로 확장할 수 있습니다.

## 11. Schedule

진행 과정은 먼저 AI Hub 데이터 구조를 분석하고, MP4와 XML 라벨 구조를 확인했습니다.

이후 XML 파싱, 행동 구간 추출, 프레임 샘플링, Keypoint 정규화까지 전처리 파이프라인을 구축했습니다.

그 다음 RGB Average Pooling, CNN-GRU, Keypoint baseline, Fusion 모델을 순서대로 구현하고 성능을 비교했습니다.

현재는 결과표, Confusion Matrix, Learning Curve까지 정리한 상태입니다.

## 12. 역할 분담

역할은 데이터 전처리, RGB 모델, Keypoint 및 Fusion 모델로 나누어 진행했습니다.

이준희 팀원은 XML 구조 분석, 라벨 파싱, 영상 전처리, Keypoint 추출, 데이터셋 구성을 담당했습니다.

정혜림 팀원은 ResNet18 기반 RGB feature 추출, Average Pooling 모델, CNN-GRU 모델 구현과 평가를 담당했습니다.

최기서 팀원은 Keypoint 정규화, 1D-CNN과 GRU 기반 Keypoint 모델, RGB와 Keypoint Fusion 모델 구현 및 성능 비교를 담당했습니다.

## 13. Q&A

이상으로 CCTV 영상과 사람 포즈 정보를 결합한 멀티모달 이상행동 분류 실험 발표를 마치겠습니다.

질문 있으시면 답변드리겠습니다.
