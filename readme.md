
# 배터리 상태 예측 및 주행 보조 Vertical AI 모델 개발 : Formula Student 차량을 위한 시계열 데이터 활용
> 2025 Formula Student Korea (FSK) — Technical Idea 

Formula Student 차량의 주행 중 배터리 상태를 실시간으로 예측하는 시스템을 개발했다.
GRPA580140 파우치형 배터리 셀의 실험 데이터를 기반으로, SOC·온도·가용 출력(SoP)을 동시에 예측하는 멀티태스크 딥러닝 파이프라인을 구축했다.

---

## 문제 정의

Formula 주행 환경에서는 짧은 시간 내에 배터리 상태를 파악하고 주행 전략을 결정해야 한다. 그러나 기존 시스템은 센서 측정값을 단순 모니터링 하는 것에 그쳐 미래 상태 예측이 부족하다. 

배터리 셀 온도, Soc, 주행 가능 거리, 가용 출력, 임계 온도 도달 시간을 실시간으로 예측해 주행 전략을 최적화하는 것을 목표로 한다.
이를 통해 팀장과 운전자가 주행 중 상태 변화를 사전에 파악하고 상황에 맞는 의사결정을 신속히 내리도록 지원한다.

---

## 데이터

- 실험 대상: GRPA580140 pouch cell (Formula Student 차량 탑재 모델)
- 측정 항목: 전압, 전류, 온도(Busbar/TC), SOC, 누적 에너지(Wh), SoC별 내부저항
- 샘플링 주기: 100ms
- 주행 사이클: OptimumLap 시뮬레이션 기반 전력 프로파일 → Simulink → 사이클러 테스트 적용

데이터 소스는 3종 (수집 주기·종료 시점 불일치):

| 소스 | 주요 변수 |
|------|-----------|
| 사이클러 | Current, Voltage, Power, Capacity, SOC |
| BMS | Busbar Temp 1/2/3 |
| TC  | TCTemp3 (셀 표면 온도) |

---

## 전처리 파이프라인

### Step 1. 리샘플링 

BMS·TC·사이클러 데이터의 수집 주기가 달라 MATLAB을 활용해 0.1s 간격 선형 보간 후 공통 시간 축으로 정렬한다.

### Step 2. 데이터 병합

리샘플링된 BMS·TC·사이클러 파일을 공통 시간 축 기준으로 병합한다.  
중복 타임스탬프 제거 및 공통 구간만 추출한다.

### Step 3. 전처리 

- 모든 변수에 값이 존재하는 마지막 행까지만 유지, 이후 결측 구간 삭제
- EWMA 기반 잔차 계산 → robust z-score 초과 구간을 이상치로 판정 → 선형 보간

정상성 변환은 전 변수에 일괄 적용하지 않고 선택적 적용:

| 변환 | 적용 대상 | 목적 |
|------|-----------|------|
| Yeo-Johnson | Current, Power | 음수 포함 분포 정규화 |
| 1차 차분 | Current, Power | 장기 추세 제거 |
| 미적용 | Voltage, SOC, Temperature | 물리적으로 제한된 범위, 변동 안정적 |

스케일링: **RobustScaler** (중앙값 + IQR 기준) — 학습 데이터만으로 fit 후 전체 적용

### Step 4. 윈도우 슬라이싱 

| 항목 | 값 |
|------|-----|
| 샘플링 간격(dt) | 0.25s |
| 입력 길이 | 60s → 240 steps |
| 예측 길이 | 30s → 120 steps |
| 입력 텐서 형태 | (N, 240, features) |
| 세션 경계 | 동일 세션 내부에서만 슬라이딩 |

### Step 5. 데이터셋 분할 

| 분할 | 비율 |
|------|------|
| Train | 65% |
| Validation | 10% |
| Test | 25% |

---

## 모델 아키텍처

3가지 시계열 백본을 공통 멀티태스크 헤드(`heads.py`)에 연결하여 비교:

| 파일 | 모델 | 특징 |
|------|------|------|
| `lstm.py` | LSTM | 장기 의존성 학습, 순환 구조 |
| `tcn.py` | TCN | 1D causal + dilated conv, 병렬 처리, residual block |
| `transformer.py` | Transformer | Self-attention 기반 전 구간 상관관계 학습 |

**예측 타깃 (Multi-task Regression):**
- SOC (State of Charge)
- Temperature high / avg / low
- P_cell (가용 셀 출력, kW)

**학습 설정:**
- Optimizer: Adam
- Loss: Smooth L1 기반 멀티태스크 가중합 (SOC 1.0, Temp 0.33, P_cell 1.0)
- Gradient Clipping (norm=1.0)
- AMP (Automatic Mixed Precision)
- Checkpoint: validation MAE 최소 지점

---

## 하이퍼파라미터 탐색

Optuna TPESampler 사용, 모델별 50회 trial (총 150회):

**최적 결과 (MAE 기준 상위 1개):**

| 모델 | batch_size | lr | Total MAE |
|------|-----------|-----|-----------|
| LSTM | 512 | 4.72e-6 | 0.3285 |
| TCN | 64 | 5.82e-6 | 0.3369 |
| Transformer | 512 | 5.42e-2 | 0.2331 |

---

## 결과

| 모델 | SOC MAE | Temp MAE | P_avail MAE | Total MAE | Latency (ms) |
|------|---------|----------|-------------|-----------|--------------|
| LSTM | 0.3938 | 0.1665 | 0.0122 | 0.5709 | 0.050 |
| TCN | 0.3406 | 0.0556 | 0.0047 | 0.4004 | 0.060 |
| Transformer | 0.2745 | 0.0784 | 0.0016 | **0.3537** | 0.056 |

Transformer가 Total MAE 기준 최고 성능. TCN은 GPU 메모리(6.7MB)와 MAE의 균형이 우수하여 경량 배포 시 대안.

---

## 텔레메트리 시스템

예측 결과를 실시간 대시보드로 전달하는 클라우드 아키텍처:

```
차량 BMS → AWS IoT Core (MQTT) → EC2 (전처리 + 추론) → WebSocket → 대시보드
```

예측 항목: SOC, 온도(최고/평균/최저), SoP, 예상 주행 가능 거리, 임계 온도 도달까지 남은 시간

---

## 디렉토리 구조

```
.
├── assets/
│   └── poster.pdf
├── matlab/
│   ├── interpolate_bms.m          # BMS 데이터 0.1s 보간
│   ├── interpolate_tctemp.m       # TC 온도 데이터 보간
│   └── interpolate_tctemp_var.m   # TC 온도 (가변 시간축) 보간
├── notebooks/
│   ├── 01_BMS_data_combining.ipynb
│   └── 02_preprocessing_summarized.ipynb
├── src/
│   ├── models/
│   │   ├── lstm.py
│   │   ├── tcn.py
│   │   ├── transformer.py
│   │   ├── heads.py
│   │   ├── registry.py
│   │   └── __init__.py
│   ├── preprocess.py
│   ├── labels.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   └── config.py
├── data/
│   ├── raw/          # 원본 CSV (gitignore)
│   ├── interim/      # 정제·병합 데이터
│   └── processed/    # 윈도우 데이터셋 (.npz)
├── requirements.txt
└── README.md
```
