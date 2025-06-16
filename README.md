# Deep Learning Experiment Framework

이 프로젝트는 Hydra와 WandB를 활용한 딥러닝 실험 환경입니다. 여러 아키텍처를 체계적으로 실험하고 결과를 추적할 수 있습니다.

## 프로젝트 구조

```
DeepPro/
├── configs/                 # Hydra 설정 파일
│   ├── config.yaml         # 메인 설정
│   ├── model/              # 모델 설정
│   │   ├── baseline.yaml
│   │   ├── se_resnet.yaml
│   │   └── inception_resnet.yaml
│   ├── optimizer/          # 옵티마이저 설정
│   │   ├── adam.yaml
│   │   └── sgd.yaml
│   └── scheduler/          # 스케줄러 설정
│       ├── step_lr.yaml
│       └── cosine.yaml
├── models/                 # 모델 클래스
│   ├── __init__.py
│   ├── baseline.py
│   ├── se_resnet.py
│   └── inception_resnet.py
├── utils/                  # 유틸리티
│   ├── logger.py          # WandB 로깅
│   └── data_utils.py      # 데이터 처리
├── train.py               # 메인 훈련 스크립트
├── requirements.txt       # 의존성
└── README.md
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 1. 기본 실행
```bash
python train.py
```

### 2. 다른 모델로 실험
```bash
# SE-ResNet으로 실험
python train.py model=se_resnet

# Inception-ResNet으로 실험  
python train.py model=inception_resnet

# Baseline 모델로 실험
python train.py model=baseline
```

### 3. 옵티마이저 변경
```bash
# SGD 옵티마이저 사용
python train.py optimizer=sgd

# Adam 옵티마이저 사용 (기본값)
python train.py optimizer=adam
```

### 4. 스케줄러 변경
```bash
# Cosine Annealing 스케줄러 사용
python train.py scheduler=cosine

# Step LR 스케줄러 사용 (기본값)
python train.py scheduler=step_lr
```

### 5. 하이퍼파라미터 오버라이드
```bash
# 학습률 변경
python train.py optimizer.lr=0.001

# 배치 사이즈 변경
python train.py data.batch_size=128

# 에포크 수 변경
python train.py experiment.num_epochs=100
```

### 6. 조합 실험
```bash
# SE-ResNet + SGD + Cosine 스케줄러
python train.py model=se_resnet optimizer=sgd scheduler=cosine

# 하이퍼파라미터와 함께
python train.py model=inception_resnet optimizer.lr=0.001 data.batch_size=128
```

### 7. WandB 없이 실행
```bash
python train.py logging.use_wandb=false
```

## 실험 예시

### 다양한 모델 비교 실험
```bash
# 실험 1: SE-ResNet
python train.py model=se_resnet experiment.name="se_resnet_exp"

# 실험 2: Inception-ResNet  
python train.py model=inception_resnet experiment.name="inception_resnet_exp"

# 실험 3: Baseline
python train.py model=baseline experiment.name="baseline_exp"
```

### 하이퍼파라미터 튜닝
```bash
# 학습률 실험
python train.py optimizer.lr=0.01 experiment.name="lr_0.01"
python train.py optimizer.lr=0.005 experiment.name="lr_0.005"
python train.py optimizer.lr=0.001 experiment.name="lr_0.001"

# 배치 사이즈 실험
python train.py data.batch_size=64 experiment.name="bs_64"
python train.py data.batch_size=128 experiment.name="bs_128"
python train.py data.batch_size=256 experiment.name="bs_256"
```

## 모델 아키텍처

### 1. Baseline Model
- 기본 ResNet 스타일 아키텍처
- Skip connection 사용

### 2. SE-ResNet Model  
- **ResNet + SENet 융합**
- Squeeze-and-Excitation 모듈로 채널 어텐션 적용
- 더 나은 특징 표현 학습

### 3. Inception-ResNet Model
- **Inception + ResNet 융합** 
- 다양한 크기의 필터를 병렬로 사용
- 계산 효율성과 표현력 향상

## WandB 로깅

프로젝트는 자동으로 다음을 WandB에 로깅합니다:
- 훈련/검증 정확도 및 손실
- 모델 파라미터 수
- 하이퍼파라미터
- 모델 가중치 (선택적)
- 모델 아티팩트

## 출력 파일

- `checkpoints/model_best.pt`: 최고 성능 모델
- `checkpoints/model_final.pt`: 최종 모델
- Hydra 로그: `outputs/YYYY-MM-DD/HH-MM-SS/`

## 주의사항

- 시드가 고정되어 재현 가능합니다 (seed=777)
- GPU 사용 시 자동으로 CUDA 활용
- 모든 설정은 Hydra를 통해 관리됩니다