# 2022 Samsung AI Challenge (3D Metrology) baseline

SEM 이미지(72x48, 흑백)로부터 깊이 맵을 예측하는 U-Net 베이스라인입니다.
평가 지표는 0-255 픽셀 스케일의 RMSE입니다.

## 설치

```bash
pip install -r requirements.txt
```

## 데이터 배치

Dacon에서 받은 zip을 풀어 아래 구조로 둡니다 (`data/`는 git에 올라가지 않습니다).

```
data/
  train/SEM/*.png
  train/Depth/*.png
  simulation_data/Case_1 ~ Case_4/{SEM,Depth}/*.png
  test/SEM/*.png
```

SEM 파일명 끝의 `_itrN`을 떼어낸 이름으로 Depth 파일과 짝을 맞춥니다.

## 학습

```bash
python train.py --data-root ./data --epochs 30 --batch-size 64 --amp
```

- 검증 세트는 깊이 맵 단위로 분리합니다. 같은 깊이 맵의 SEM 4장이 train/val에 나뉘어 들어가지 않습니다.
- `--sim-cases 1,2,3,4` 로 시뮬레이션 데이터를 학습에 추가할 수 있습니다.
- `--limit 200` 으로 작은 데이터로 빠르게 동작만 확인할 수 있습니다.
- 결과: `runs/baseline/best.pt`, `last.pt`, `history.json`.

## 추론과 제출 파일

```bash
python predict.py --data-root ./data --checkpoint runs/baseline/best.pt --out submission.zip --tta
```

`submission.zip` 안에 test SEM과 같은 파일명의 depth png가 들어갑니다.

## 동작 확인 (데이터 없이)

```bash
python tests/smoke_test.py
```

가짜 이미지를 만들어 학습 1 epoch과 추론, zip 생성까지 실행합니다.
