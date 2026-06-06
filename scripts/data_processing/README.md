# Data Processing Scripts

AI Hub 원본 데이터를 학습 가능한 형태로 가공하는 스크립트 모음입니다.

## 실행 순서

```powershell
.\.venv5070\Scripts\python.exe scripts\data_processing\extract_aihub_dataset.py
.\.venv5070\Scripts\python.exe scripts\data_processing\build_manifest.py
.\.venv5070\Scripts\python.exe scripts\data_processing\inspect_video_fps.py
.\.venv5070\Scripts\python.exe scripts\data_processing\extract_frames_224.py
.\.venv5070\Scripts\python.exe scripts\data_processing\build_train_val_test_manifest.py
```

## 산출물

```text
data/extracted/
data/processed/frames_224_manifest.csv
data/processed/frame_index_224.csv
data/processed/frames_224_trainvaltest.csv
```
