from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from PIL import Image, ImageEnhance
import mplfinance as mpf
import os
import io
from typing import List, Dict, Optional

# ============================================================
# FastAPI 앱 생성
# ============================================================

app = FastAPI(
    title="Patron API",
    description="차트 패턴 유사도 검색 시스템",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 경로 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models/best_model.pth")
METADATA_PATH = os.path.join(BASE_DIR, "data/metadata_all.csv")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data/raw")


# ============================================================
# Pydantic 모델
# ============================================================
class PatronSearchRequest(BaseModel):
    ticker: str
    date: str  # YYYY-MM-DD

class OHLCData(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float

class QueryInfo(BaseModel):
    ticker: str
    date: str
    sector: str
    ohlc_data: Optional[List[OHLCData]] = None  # 쿼리 OHLC 데이터 (선택적)

class Top3Item(BaseModel):
    rank: int
    ticker: str
    date: str
    sector: str
    distance: float
    similarity_percent: float
    ohlc_data: List[OHLCData]
    returns: Dict[str, Optional[float]]

class PatronSearchResponse(BaseModel):
    query: QueryInfo  # Dict에서 QueryInfo 모델로 변경
    top3: List[Top3Item]

# ============================================================
# ResNet18 모델 클래스
# ============================================================
class ChartEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=512):
        super(ChartEmbeddingModel, self).__init__()
        resnet = models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.embedding_dim = embedding_dim

    def forward(self, x):
        features = self.features(x)
        embeddings = features.view(features.size(0), -1)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

# ============================================================
# 전역 변수
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
metadata = None
embeddings = None
faiss_index = None 
US_EXCHANGES = ['NMS', 'NYSE', 'NYSEARCA', 'NYSEMKT', 'NYQ', 'NGM', 'NCM']

# ============================================================
# 서버 시작 시 모델 및 데이터 로드
# ============================================================
@app.on_event("startup")
async def startup_event():
    global model, metadata, embeddings, faiss_index

    print("=" * 60)
    print("🚀 Patron FastAPI 서버 시작")
    print("=" * 60)
    print(f"디바이스: {device}")

    # 1. 모델 로드
    print("\n📥 모델 로딩 중...")
    model = ChartEmbeddingModel(embedding_dim=512).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ 모델 로드 완료 (Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.6f})")

    # 2. 메타데이터 로드
    print("\n📊 메타데이터 로딩 중...")
    metadata = pd.read_csv(METADATA_PATH)
    metadata = metadata.dropna(subset=['return_3m'])
    metadata = metadata.reset_index(drop=True)
    print(f"✅ 메타데이터 로드 완료: {len(metadata):,}개 패턴")

    # 3. Faiss 인덱스 로드
    print("\n🔍 Faiss 인덱스 로딩 중...")
    import faiss
    FAISS_INDEX_PATH = os.path.join(BASE_DIR, "data/faiss_index.bin")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"✅ Faiss 인덱스 로드 완료: {faiss_index.ntotal:,}개 벡터")

    # 4. 임베딩 로드
    print("\n📦 임베딩 로딩 중...")
    EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data/embeddings.npy")
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"✅ 임베딩 로드 완료: {embeddings.shape}")

    print("\n" + "=" * 60)
    print("✅ 서버 준비 완료!")
    print(f"📡 API 문서: http://localhost:8000/docs")
    print("=" * 60)

# ============================================================
# 유틸리티 함수들
# ============================================================

def validate_us_stock(ticker: str) -> tuple:
    """미국 주식 여부 검증"""
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info

        if not info or len(info) < 5:
            return False, "No Data"

        exchange = info.get('exchange', 'Unknown')

        if exchange in US_EXCHANGES:
            return True, exchange
        else:
            return False, exchange
    except Exception as e:
        return False, f"Error: {str(e)[:30]}"

def ohlc_to_grayscale_image(ohlc_array: np.ndarray, temp_path='/tmp/temp_chart.png') -> np.ndarray:
    """
    (12, 4) OHLC numpy 배열 → (224, 224) 그레이스케일 numpy 배열
    노트북 01 전처리와 100% 동일한 방식
    """
    try:
        df = pd.DataFrame(ohlc_array, columns=['Open', 'High', 'Low', 'Close'])
        df.index = pd.date_range('2020-01-01', periods=len(df), freq='W')

        mc = mpf.make_marketcolors(
            up='white', down='white',
            edge='white', wick='white',
            volume='white'
        )

        s = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='',
            y_on_right=False,
            facecolor='black',
            figcolor='black'
        )

        mpf.plot(
            df, type='candle', style=s,
            savefig=temp_path,
            figsize=(2.24, 2.24),
            axisoff=True,
            closefig=True
        )

        img = Image.open(temp_path).convert('L')
        img_resized = img.resize((224, 224), Image.LANCZOS)
        enhancer = ImageEnhance.Contrast(img_resized)
        img_enhanced = enhancer.enhance(1.5)

        img_array = np.array(img_enhanced, dtype=np.uint8)

        img.close()
        img_resized.close()

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return img_array

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 생성 실패: {str(e)}")

def get_top3_diverse_exclude_self(
    query_embedding: np.ndarray,
    query_ticker: str,
    query_date: str,
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    k: int = 100
) -> List[Dict]:
    """
    Top-3 탐색 알고리즘
    - L2 Distance 기반
    - 본인 제외 (같은 종목 + 14일 이내)
    - 종목 중복 제거

    Args:
        query_embedding: 쿼리 임베딩 (512,)
        query_ticker: 쿼리 종목 코드
        query_date: 쿼리 기준 날짜 (YYYY-MM-DD)
        metadata: 메타데이터 DataFrame
        embeddings: 전체 임베딩 (N, 512)
        k: 후보 개수 (기본 100)

    Returns:
        top3: 상위 3개 패턴 정보
    """
    # 1. L2 Distance 계산 (작을수록 유사)
    distances = np.linalg.norm(embeddings - query_embedding, axis=1)

    # 2. Top-K 후보 선택 (거리 작은 순)
    top_k_indices = np.argsort(distances)[:k]
    top_k_distances = distances[top_k_indices]

    # 3. 본인 제외 (같은 종목 + 14일 이내만 제외)
    selected = []
    seen_tickers = set()
    query_date_dt = pd.to_datetime(query_date)

    for i in range(k):
        idx = int(top_k_indices[i])
        ticker = metadata.loc[idx, 'ticker']
        start_date = pd.to_datetime(metadata.loc[idx, 'start_date'])

        # 본인 제외 (같은 종목 + 14일 이내)
        if ticker == query_ticker:
            date_diff_days = abs((start_date - query_date_dt).days)
            if date_diff_days < 14:
                continue

        if ticker in seen_tickers:
            continue
            
        distance = top_k_distances[i]

        selected.append({
            'idx': idx,
            'ticker': ticker,
            'pattern_id': int(metadata.loc[idx, 'pattern_id']),
            'start_date': metadata.loc[idx, 'start_date'],
            'end_date': metadata.loc[idx, 'end_date'],
            'date': metadata.loc[idx, 'start_date'],
            'distance': distance,
            'sector': metadata.loc[idx, 'sector'],
            'return_3m': float(metadata.loc[idx, 'return_3m']),
            'return_6m': float(metadata.loc[idx, 'return_6m']) if pd.notna(metadata.loc[idx, 'return_6m']) else None,
            'return_1y': float(metadata.loc[idx, 'return_1y']) if pd.notna(metadata.loc[idx, 'return_1y']) else None
        })

        # Top-3 완성
        if len(selected) == 3:
            break

    return selected

# ============================================================
# API 엔드포인트
# ============================================================

@app.post("/api/patron/search", response_model=PatronSearchResponse)
async def patron_search(request: PatronSearchRequest):
    """
    차트 패턴 유사도 검색
    
    Args:
        request: 종목 코드(ticker) + 날짜(date)
    
    Returns:
        Top-3 유사 패턴 + OHLC 데이터
    """
    ticker = request.ticker.upper()
    date_str = request.date

    print(f"\n{'='*60}")
    print(f"🔍 검색 요청: {ticker} ({date_str})")
    print(f"{'='*60}")

    # 1. 미국 주식 검증
    print(f"📋 Step 1: 미국 주식 검증 중...")
    is_valid, exchange = validate_us_stock(ticker)

    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"INVALID_TICKER: 미국 주식이 아닙니다 (거래소: {exchange})"
        )

    print(f"✅ {ticker}는 미국 주식입니다 (거래소: {exchange})")

    # 섹터 정보 가져오기
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    query_sector = info.get('sector', 'Unknown')

    # 2. yfinance에서 12주 데이터 다운로드
    print(f"\n📥 Step 2: yfinance 데이터 다운로드 중...")
    try:
        end_date = pd.to_datetime(date_str)
        start_date = end_date - timedelta(weeks=20)

        data = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1wk',
            progress=False,
            auto_adjust=True
        )

        if data.empty:
            raise HTTPException(status_code=404, detail="데이터가 없습니다")

        # OHLC만 추출
        ohlc_data = data[['Open', 'High', 'Low', 'Close']].copy()

        # 멀티인덱스 제거
        if isinstance(ohlc_data.columns, pd.MultiIndex):
            ohlc_data.columns = ['Open', 'High', 'Low', 'Close']

        # 마지막 12주만 선택
        last_12_weeks = ohlc_data.tail(12)

        if len(last_12_weeks) < 12:
            raise HTTPException(
                status_code=400,
                detail=f"데이터 부족 (현재: {len(last_12_weeks)}주, 필요: 12주)"
            )

        print(f"✅ 다운로드 완료: {len(last_12_weeks)}주")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"yfinance 다운로드 실패: {str(e)}")

    # 3. MinMaxScaler 정규화
    print(f"\n🔄 Step 3: MinMaxScaler 정규화 중...")
    scaler = MinMaxScaler()
    normalized_ohlc = scaler.fit_transform(last_12_weeks.values)
    print(f"✅ 정규화 완료: {normalized_ohlc.shape}")

    # 4. 이미지 생성
    print(f"\n🎨 Step 4: 차트 이미지 생성 중...")
    query_img = ohlc_to_grayscale_image(normalized_ohlc)
    print(f"✅ 이미지 생성 완료: {query_img.shape}")

    # 5. 임베딩 추출
    print(f"\n🔥 Step 5: 임베딩 추출 중...")
    with torch.no_grad():
        query_img_tensor = torch.FloatTensor(query_img).unsqueeze(0).unsqueeze(0).to(device) / 255.0
        query_embedding = model(query_img_tensor).squeeze(0).cpu().numpy()
    print(f"✅ 임베딩 추출 완료: {query_embedding.shape}")

    # 6. Top-3 검색
    print(f"\n🔍 Step 6: Top-3 유사 패턴 검색 중...")
    top3_results = get_top3_diverse_exclude_self(
        query_embedding=query_embedding,
        query_ticker=ticker,
        query_date=last_12_weeks.index[-1].strftime('%Y-%m-%d'),
        metadata=metadata,
        embeddings=embeddings,
        k=100
    )

    if len(top3_results) < 3:
        raise HTTPException(
            status_code=404,
            detail=f"유사 패턴을 찾을 수 없습니다 (발견: {len(top3_results)}개)"
        )

    print(f"✅ Top-3 검색 완료")

    # 7. OHLC 데이터 추가 (raw 폴더에서)
    print(f"\n📊 Step 7: OHLC 데이터 로딩 중...")
    top3_items = []

    for rank, item in enumerate(top3_results, 1):
        result_ticker = item['ticker']
        start_date = item['start_date']
        end_date = item['end_date']

        # raw CSV 로드
        try:
            csv_path = os.path.join(RAW_DATA_PATH, f"{result_ticker}.csv")
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df_period = df.loc[start_date:end_date]

            ohlc_list = []
            for date, row in df_period.iterrows():
                ohlc_list.append(OHLCData(
                    date=date.strftime('%Y-%m-%d'),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close'])
                ))

            # 유사도 퍼센트 계산 (거리 → 유사도)
            similarity_percent = 1 / (1 + item['distance']) * 100

            top3_items.append(Top3Item(
                rank=rank,
                ticker=result_ticker,
                date=start_date,
                sector=item['sector'],
                distance=item['distance'],
                similarity_percent=round(similarity_percent, 1),
                ohlc_data=ohlc_list,
                returns={
                    '3m': item['return_3m'],
                    '6m': item['return_6m'],
                    '1y': item['return_1y']
                }
            ))

        except Exception as e:
            print(f"⚠️ {result_ticker} OHLC 로드 실패: {e}")
            continue

    print(f"✅ OHLC 데이터 로딩 완료")

    # 7-1. 쿼리 종목의 OHLC 데이터 추가
    print(f"\n📊 Step 7-1: 쿼리 종목 OHLC 데이터 변환 중...")
    query_ohlc_list = []
    for date, row in last_12_weeks.iterrows():
        query_ohlc_list.append(OHLCData(
            date=date.strftime('%Y-%m-%d'),
            open=float(row['Open']),
            high=float(row['High']),
            low=float(row['Low']),
            close=float(row['Close'])
        ))
    print(f"✅ 쿼리 OHLC 데이터 변환 완료: {len(query_ohlc_list)}개")

    # 8. 응답 생성
    response = PatronSearchResponse(
        query=QueryInfo(
            ticker=ticker,
            date=date_str,
            sector=query_sector,
            ohlc_data=query_ohlc_list  # 🔥 쿼리 OHLC 데이터 추가
        ),
        top3=top3_items
    )

    print(f"\n{'='*60}")
    print(f"✅ 검색 완료!")
    print(f"{'='*60}\n")

    return response