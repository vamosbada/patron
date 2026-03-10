# Patron — Architecture & Experiment Log

## 목차

1. 파이프라인 개요
2. 데이터 수집 및 전처리
3. AI 모델 아키텍처
4. FAISS 검색 및 Top-3 알고리즘
5. FastAPI 서버
6. 실험 로그

---

## 1. 파이프라인 개요

```
yfinance (172개 종목, 주봉 OHLC)
        ↓
슬라이딩 윈도우 (12주) + MinMaxScaler 정규화
        ↓
그레이스케일 캔들스틱 이미지 (224×224)
        ↓
ResNet18 → 512차원 L2 정규화 임베딩
        ↓
Triplet Loss + Semi-hard Negative Mining 학습
        ↓
FAISS IndexFlatL2 (약 50,000개 벡터)
        ↓
FastAPI: 종목 코드 입력 → Top-3 유사 패턴 반환
```

---

## 2. 데이터 수집 및 전처리

### 데이터셋

- **출처**: yfinance — NASDAQ 100 + S&P 100 합집합, 중복 제거 → 172개 종목
- **기간**: 2020-01-01 ~ 2025-10 (주봉 OHLC, `auto_adjust=True`로 액면분할 보정)
- **패턴**: 12주 슬라이딩 윈도우 → 약 50,000개 패턴
- **메타데이터**: ticker, sector, industry, start/end date, 3·6·12개월 이후 수익률

### 전처리 과정

```
Step 1: 172개 종목 CSV 다운로드 (yfinance, 주봉)
Step 2: 12주 슬라이딩 윈도우로 패턴 생성
Step 3: 패턴별 MinMaxScaler (OHLC 4열 동시, 0~1 정규화)
Step 4: mplfinance로 224×224 그레이스케일 캔들스틱 이미지 생성
Step 5: 대비 1.5× 강화 (Pillow ImageEnhance)
Step 6: .npy 파일로 저장
```

### 정규화 방식: MinMaxScaler (패턴별)

각 12주 패턴의 OHLC 값 전체를 하나의 MinMaxScaler로 0~1 정규화한다. 절대 가격이 아닌 차트의 상대적 모양을 학습하기 위해서다. Log 정규화 대비 MinMaxScaler를 선택한 근거는 실험 로그(Experiment 2, 3) 참고.

---

## 3. AI 모델 아키텍처

### 백본: ResNet18

```python
class ChartEmbeddingModel(nn.Module):
    # ImageNet 사전학습 ResNet18
    # conv1: 3채널 → 1채널 (그레이스케일 입력)
    # FC 레이어 제거 → 512차원 출력
    # L2 정규화 (단위구 위에 임베딩)
```

- 입력: (224, 224, 1) 그레이스케일 이미지
- 출력: 512차원 L2 정규화 벡터

ResNet18을 선택한 이유: 분류 문제가 아니라 공간적 패턴 인식이 목적이므로 경량 CNN으로 충분하고, ImageNet 사전학습 가중치가 엣지/곡선 특징 추출에 유리하다.

### 학습 방식: Triplet Loss + Semi-hard Negative Mining

```
Loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
margin = 0.2
```

**Anchor-Positive 쌍**: 같은 종목의 t주차 패턴과 t+1, t+2, t+3주차 패턴을 묶는다. 시간적으로 연속된 같은 종목 차트는 유사한 패턴을 형성한다는 가정.

**Negative 선택 전략**:
- Epoch 1~2: Random Negative (다른 종목, 반대 방향 수익률)
- Epoch 3: Semi-hard Negative Mining

```
Semi-hard 조건: d(anchor, positive) < d(anchor, negative) < d(anchor, positive) + margin
```

Easy Negative는 gradient가 0에 가까워 학습 기여가 없고, Hard Negative는 학습 초기에 불안정하다. Semi-hard는 모델이 아직 구분 못하는 적당히 어려운 케이스를 선택해 학습 효율을 최대화한다.

### 학습 결과 (최종 채택 모델)

| Epoch | Train Loss | Val Loss | 비고 |
|-------|-----------|---------|------|
| 1 | 0.007197 | 0.005817 | Random Negative |
| 2 | 0.004771 | **0.005174** | Random Negative — Best |
| 3 | 0.004488 | 0.005999 | Semi-hard → Early Stop |

- 총 학습 시간: 1.5시간 (NVIDIA A100, Mixed Precision)
- 이미지 생성 포함 전체: 6.5시간

---

## 4. FAISS 검색 및 Top-3 알고리즘

### FAISS 인덱스

```python
index = faiss.IndexFlatL2(512)  # L2 거리 기반 정확 검색
index.add(embeddings)           # 약 50,000개 벡터 추가
```

### Top-3 탐색 알고리즘

```python
def get_top3(query_embedding, all_embeddings, metadata, k=100):
    # 1. L2 거리 기준 Top-100 후보 선택
    distances = np.linalg.norm(all_embeddings - query_embedding, axis=1)
    top_k_indices = np.argsort(distances)[:k]

    # 2. 쿼리 본인 제외 (같은 ticker + 14일 이내)
    # 3. 종목(ticker) 중복 제거 — 종목당 가장 유사한 패턴 1개만
    # 4. Top-3 반환
    selected = []
    seen_tickers = set()
    for idx in top_k_indices:
        ticker = metadata.loc[idx, 'ticker']
        if ticker == query_ticker and date_diff < 14:
            continue
        if ticker in seen_tickers:
            continue
        selected.append(idx)
        seen_tickers.add(ticker)
        if len(selected) == 3:
            break
    return selected
```

종목 중복 제거를 적용하는 이유: 같은 종목의 연속된 패턴이 Top-10을 독점하는 현상이 발생한다. 중복 제거 없이는 평균 2.2개의 동일 종목이 Top-3에 포함된다.

---

## 5. FastAPI 서버

**엔드포인트**: `POST /api/patron/search`

**처리 흐름**:
```
1. ticker 입력 → yfinance로 미국 거래소 검증
   (허용: NMS, NYSE, NYSEARCA, NYSEMKT, NYQ, NGM)
2. 최근 12주 OHLC 실시간 다운로드
3. MinMaxScaler 정규화 → 224×224 그레이스케일 이미지 생성
4. ResNet18로 512차원 임베딩 추출
5. FAISS 인덱스 검색 → Top-3 (종목 중복 제거, 본인 제외)
6. 메타데이터(섹터, 날짜, 3·6·12개월 수익률) 반환
```

서버 시작 시 `embeddings.npy` (사전 계산된 50k 임베딩)와 `faiss_index.bin`을 로드한다. 요청마다 임베딩을 새로 생성하면 3분이 걸리므로 사전 계산이 필수다.

---

## 6. 실험 로그

### Experiment 1 — MinMaxScaler 기반 학습 (2025-11-10)

| 항목 | 값 |
|------|----|
| 정규화 | MinMaxScaler (패턴별) |
| Anchor-Positive 쌍 | +1, +2, +3주 (113,776 train / 28,445 val) |
| Negative Mining | Epoch 1-2 Random, Epoch 3 Semi-hard |
| Batch / Epochs | 128 / 3 (Early Stop) |
| Best Val Loss | 0.005174 (Epoch 2) |

**발견된 문제**: Top-10 결과에 같은 종목이 평균 2.2개 중복됨 → 검색 시 종목 중복 제거로 해결

---

### Experiment 2 — 정규화 방법 비교 분석 (2025-11-12)

MinMaxScaler의 이론적 한계를 분석하고 4가지 정규화를 비교했다.

**문제 제기**: MinMaxScaler는 `1→6` (600% 상승)과 `10→58` (580% 상승)을 동일한 패턴으로 인식한다.

| 방법 | 같은 모양 인식 | 범위 | 극단값 처리 |
|------|-------------|------|-----------|
| MinMaxScaler | ❌ | [0, 1] | ❌ |
| First-Close 기준 | ✅ | [0, ∞] | ❌ |
| First-Close + Clip | ✅ | [0, 1] | △ |
| **Log-Base** | ✅ | [0, 1] | ✅ |

이론적으로는 Log 정규화(`log(P_t / P_0)`)가 금융 표준이자 가장 우수하다고 판단 → Experiment 3에서 실제 학습에 적용.

---

### Experiment 3 — Log 정규화 학습 (2025-11-13) → 기각

Experiment 2의 결론을 적용해 Log 정규화로 재학습.

| 항목 | 값 |
|------|----|
| 정규화 | Log-Base (`log(P_t / P_0)`) |
| Anchor-Positive 쌍 | +6, +12주 (74,475 train / 18,619 val) |
| Best Val Loss | 0.138237 (Epoch 3) — Exp 1 대비 27배 높음 |
| 학습 시간 | 4.2시간 (Epoch 6에서 Early Stop) |

**시각적 비교 결과** (랜덤 쿼리 20개, `07_visual_comparison.ipynb`):

| 지표 | Exp 1 (MinMaxScaler) | Exp 3 (Log) |
|------|---------------------|-------------|
| 평균 중복 종목 수 | 2.20개 | **0.15개** |
| 평균 L2 거리 | 0.196 | **0.061** |
| 시각적 유사도 | **눈으로 봐도 비슷함** ✅ | 숫자는 가깝지만 차트 모양이 다름 ❌ |

**Log 정규화 기각 이유**:

Log 변환은 선형 상승 패턴을 곡선으로 왜곡한다.

```
원본: [100, 110, 120, 130, 140, 150]  ← 직선 상승

MinMaxScaler: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  ← 직선 유지 ✅
Log-Base:     [0.0, 0.095, 0.182, 0.262, 0.336, 0.405]  ← 곡선으로 왜곡 ❌
```

모델이 왜곡된 모양을 학습했기 때문에 L2 거리는 가깝게 나오지만, 사람 눈에는 유사하지 않은 패턴을 검색한다. **이론과 실제의 불일치** — 숫자 지표만으로 판단하면 안 된다는 핵심 교훈.

**최종 결정**: Experiment 1 (MinMaxScaler) 채택, 중복 문제는 검색 시 종목 단위 중복 제거로 해결.

---

**최종 수정일**: 2025-11-21
**작성자**: 신바다 (Dankook University)
