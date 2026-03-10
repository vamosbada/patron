[English Version](README.md)

# Patron — 차트 패턴 유사도 검색

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> 미국 주식 종목 코드를 입력하면, 최근 12주 차트와 가장 비슷했던 과거 패턴 3개를 찾아 그 이후 주가 흐름을 보여줍니다.

*Patron*은 스페인어로 '패턴'이라는 뜻입니다.

---

## 개요

Patron은 주식 초보자를 위한 교육 도구입니다. 복잡한 기술적 지표 없이 딱 하나의 질문에 답합니다: **"이 차트 패턴, 과거에도 있었을까? 그 이후엔 어떻게 됐을까?"**

ResNet18로 차트 이미지를 512차원 벡터로 변환하고, FAISS로 약 50,000개의 과거 패턴 중 가장 유사한 것 3개(각각 다른 종목)를 찾아, 3·6·12개월 이후 수익률과 함께 반환합니다.

**이 서비스는 교육 목적이며, 투자 조언이 아닙니다.**

---

## 핵심 설계 결정

| 구성 요소 | 선택 | 이유 |
|-----------|------|------|
| 백본 모델 | ResNet18 (ImageNet 사전학습) | 경량이면서 공간 특징 추출 성능 우수 |
| 학습 방식 | Triplet Loss + Semi-hard Negative Mining | 분류가 아닌 상대적 유사도 학습 |
| 정규화 | MinMaxScaler (패턴별) | 차트 모양 보존. Log 정규화는 모양을 왜곡함 (실험 2 참고) |
| 검색 | FAISS L2 | 50,000개 벡터 빠른 정확 검색 |
| 중복 제거 | 종목 단위 | 같은 종목이 Top-3를 독점하는 현상 방지 |

가장 중요한 발견: **Log 정규화는 수치상으로는 더 좋아 보였지만, 실제 차트 모양을 왜곡**해서 숫자는 가깝지만 눈으로 보면 다른 패턴을 검색하는 문제가 발생했습니다. 결국 MinMaxScaler를 최종 채택했습니다. 전체 실험 기록은 `ARCHITECTURE.md`를 참고하세요.

---

## 데이터셋

- **출처**: [yfinance](https://github.com/ranaroussi/yfinance) — 미국 주식 172개 (NASDAQ 100 + S&P 100, 중복 제거)
- **기간**: 2020-01-01 ~ 2025-10 (주봉 OHLC, 액면분할 자동 보정)
- **패턴 수**: 약 50,000개 (12주 슬라이딩 윈도우)
- **메타데이터**: 종목, 섹터, 산업, 패턴 날짜, 3·6·12개월 이후 수익률

원본 CSV 파일 172개 (~4 MB)는 `patron_fastapi/data/raw/`에 포함되어 있습니다. 모델 가중치(`.pth`)와 이미지 배열(`.npy`, `.tar`)은 용량 문제로 Google Drive에 보관하며 **이 레포에는 포함되지 않습니다**.

---

## 노트북

처음부터 재현하려면 순서대로 실행하세요:

| # | 노트북 | 내용 |
|---|--------|------|
| 01 | `01_preprocessing.ipynb` | OHLC 데이터 수집 → 슬라이딩 윈도우 → MinMaxScaler → 그레이스케일 차트 이미지 생성 |
| 02 | `02_training_v1.ipynb` | ResNet18 + Triplet Loss 학습 (**최종 모델**, T4 기준 약 6.5시간) |
| 03 | `03_faiss_search.ipynb` | FAISS 인덱스 구축, 종목 중복 제거 Top-3 검색 구현 |
| 04 | `04_normalization_compare.ipynb` | MinMaxScaler vs Log 정규화 비교 분석 (실험 2) |
| 05 | `05_preprocessing_v2.ipynb` | Log 정규화 전처리 (실험 3 — 최종 기각) |
| 06 | `06_training_v2.ipynb` | Log 정규화 적용 학습 (실험 3 — 기각) |
| 07 | `07_visual_comparison.ipynb` | 실험 1 vs 실험 3 시각적 비교 |
| 08 | `08_realtime_search.ipynb` | yfinance 실시간 데이터로 검색 데모 |
| 09 | `09_embedding_precompute.ipynb` | FastAPI용 임베딩 50,000개 사전 계산 및 저장 |

> 노트북 04~07은 정규화 실험 기록입니다. 최종 모델 실행에는 필요 없지만, MinMaxScaler를 선택한 근거가 담겨 있습니다.

---

## FastAPI 서버

`patron_fastapi/main.py`는 단일 엔드포인트를 제공합니다:

```
POST /api/patron/search
Body: ticker (str, 예: "AAPL")
```

**처리 과정:**
1. 종목 코드를 미국 거래소 기준으로 검증 (yfinance)
2. 최근 12주 OHLC 데이터를 실시간으로 가져옴
3. 그레이스케일 차트 이미지로 변환
4. ResNet18로 512차원 임베딩 추출
5. FAISS 인덱스에서 Top-3 유사 패턴 검색 (종목 중복 제거)
6. 섹터, 날짜, 이후 수익률 메타데이터 반환

**실행 방법** (`patron_fastapi/requirements.txt` 참고):

```bash
cd patron_fastapi
pip install -r requirements.txt

# 모델 파일을 아래 경로에 배치:
# patron_fastapi/models/best_model.pth     (ResNet18 가중치)
# patron_fastapi/data/embeddings.npy       (50k 임베딩)
# patron_fastapi/data/metadata_all.csv     (패턴 메타데이터)

uvicorn main:app --host 0.0.0.0 --port 8000
```

서버 설치 가이드 및 알려진 이슈는 `patron_fastapi/SERVER.md`를 참고하세요.

---

## 실험 결과

| 실험 | 정규화 | 평균 중복 종목 수 | 평균 L2 거리 | 시각적 품질 |
|------|--------|------------------|--------------|-------------|
| 실험 1 — **최종 채택** | MinMaxScaler | 2.20개 | 0.196 | **눈으로 봐도 유사** ✅ |
| 실험 3 — 기각 | Log | 0.15개 | 0.061 | 수치는 좋지만 시각적으로 다름 ❌ |

실험 3이 수치 지표는 더 좋았지만, 실제 차트 이미지를 눈으로 비교했을 때 유사하지 않은 패턴을 검색했습니다. 실험 1 (MinMaxScaler)을 최종 모델로 채택하고, 중복 문제는 검색 시 종목 단위 중복 제거로 해결했습니다.

---

## 프로젝트 구조

```
patron/
├── notebooks/
│   ├── 01_preprocessing.ipynb        # 데이터 수집 및 전처리
│   ├── 02_training_v1.ipynb          # 최종 모델 학습 (ResNet18 + Triplet Loss)
│   ├── 03_faiss_search.ipynb         # FAISS 인덱스 + Top-3 검색
│   ├── 04_normalization_compare.ipynb # MinMaxScaler vs Log 분석
│   ├── 05_preprocessing_v2.ipynb     # Log 정규화 전처리
│   ├── 06_training_v2.ipynb          # Log 정규화 학습 (기각)
│   ├── 07_visual_comparison.ipynb    # 실험 1 vs 실험 3 시각적 비교
│   ├── 08_realtime_search.ipynb      # 실시간 yfinance 데모
│   └── 09_embedding_precompute.ipynb # FastAPI용 임베딩 사전 계산
├── patron_fastapi/
│   ├── main.py                       # FastAPI 서버
│   ├── requirements.txt
│   ├── SERVER.md                     # 설치 및 배포 가이드
│   └── data/
│       ├── metadata_all.csv          # 패턴 메타데이터 (약 50,000행)
│       └── raw/                      # 종목별 CSV 172개 (~4 MB)
├── ARCHITECTURE.md                   # 전체 설계 문서 및 실험 로그
└── .gitignore
```

**포함되지 않음** (Google Drive 보관):
- `best_model.pth` — 학습된 ResNet18 가중치
- `embeddings.npy` — 50,000개 임베딩 벡터
- `images.tar` / `images_v2.tar` — 차트 이미지 아카이브 (각 약 2.4 GB)

---

## 개발 배경

Patron은 주식 초보자를 위한 AI 교육 플랫폼 **Dipping**의 차트 패턴 검색 모듈입니다. Dipping은 단국대학교 창업동아리 QuantrumAI에서 개발했으며, **교육부 인증 U300 정부 창업 지원 프로그램**에 선정된 프로젝트입니다.

신바다가 QuantrumAI ML 엔지니어로서 2025년 7월부터 12월까지 Patron의 전체 AI 파이프라인(데이터 수집, 모델 학습, FAISS 인덱싱, FastAPI 서버)을 단독 개발했습니다.

---

## 라이선스

MIT License — [LICENSE](LICENSE) 참고
