# Patron FastAPI Server

## 개요

종목 코드를 입력하면 최근 12주 차트와 유사한 과거 패턴 Top-3를 반환하는 API 서버.

---

## 폴더 구조

```
patron_fastapi/
├── main.py                  # FastAPI 서버
├── requirements.txt
├── SERVER.md
├── models/
│   └── best_model.pth       # ResNet18 가중치 (128MB) — Google Drive에서 다운로드
└── data/
    ├── metadata_all.csv     # 패턴 메타데이터 (49,987행)
    ├── embeddings.npy       # 사전 계산된 임베딩 (97.6MB) — Google Drive에서 다운로드
    ├── faiss_index.bin      # FAISS 인덱스 (97.6MB) — Google Drive에서 다운로드
    └── raw/                 # 172개 종목 원본 OHLC CSV
```

`metadata_all.csv`와 `raw/`는 레포에 포함되어 있음. 나머지 3개 파일은 용량 문제로 Google Drive에 보관.

---

## 설치 및 실행

### 1. 대용량 파일 다운로드

아래 3개 파일을 Google Drive에서 다운로드해 지정 경로에 배치:

| 파일 | 크기 | 경로 |
|------|------|------|
| `best_model.pth` | 128MB | `patron_fastapi/models/` |
| `embeddings.npy` | 97.6MB | `patron_fastapi/data/` |
| `faiss_index.bin` | 97.6MB | `patron_fastapi/data/` |

### 2. 패키지 설치

```bash
cd patron_fastapi
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 서버 실행

```bash
uvicorn main:app --reload --port 8000
```

- 서버: `http://localhost:8000`
- API 문서: `http://localhost:8000/docs`

---

## API

### POST `/api/patron/search`

**요청:**
```json
{
  "ticker": "NVDA",
  "date": "2024-12-01"
}
```

**응답:**
```json
{
  "query": {
    "ticker": "NVDA",
    "date": "2024-12-01",
    "sector": "Technology",
    "ohlc_data": [...]
  },
  "top3": [
    {
      "rank": 1,
      "ticker": "UBER",
      "date": "2020-11-11",
      "sector": "Technology",
      "distance": 0.3757,
      "similarity_percent": 72.5,
      "ohlc_data": [...],
      "returns": { "3m": 1.24, "6m": -17.96, "1y": -31.62 }
    }
  ]
}
```

**에러 응답:**
```json
{
  "detail": "INVALID_TICKER: 미국 주식이 아닙니다 (거래소: OTCMKTS)"
}
```

---

## 제약사항

- 미국 주식만 지원 (허용 거래소: NMS, NYSE, NYSEARCA, NYSEMKT, NYQ, NGM)
- 주봉 데이터 기준 (일봉/분봉 미지원)
- 입력 날짜 기준 이전 12주 데이터가 있어야 함
- 최근 패턴은 `return_6m`, `return_1y`가 null일 수 있음
