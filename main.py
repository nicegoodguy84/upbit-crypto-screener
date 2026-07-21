import base64
import io
import os
import platform
import re
import warnings
from datetime import datetime, timedelta, timezone

import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyupbit
from tqdm import tqdm

warnings.filterwarnings("ignore")

# OS별 폰트 동적 설정 (크로스플랫폼 및 GitHub Actions 한글 깨짐 방지)
system_name = platform.system()
if system_name == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
elif system_name == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    # Linux (GitHub Actions 환경)
    plt.rcParams["font.family"] = "NanumGothic"

plt.rcParams["axes.unicode_minus"] = False


def calculate_minervini_base(df_hist):
    if len(df_hist) < 200:
        return 1

    df_hist["MA200_slope"] = df_hist["MA200"].diff(5)
    stage2_df = df_hist[df_hist["MA200_slope"] > 0]

    if len(stage2_df) < 20:
        return 1

    base_count = 1
    highest_price = stage2_df["close"].iloc[0]
    in_correction = False

    for idx, row in stage2_df.iterrows():
        price = row["close"]
        if price > highest_price:
            highest_price = price
            if in_correction:
                base_count += 1
                in_correction = False
        elif price < highest_price * 0.90:
            in_correction = True

    return min(base_count, 4)


def generate_chart_image(ticker, name, df_hist, w_point, m_point, base_stage):
    df_plot = df_hist.tail(120).copy()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(
        df_plot.index,
        df_plot["close"],
        label="현재가",
        color="#1e293b",
        linewidth=2,
    )
    ax1.plot(
        df_plot.index,
        df_plot["MA20"],
        label="20일선",
        color="#ef4444",
        linestyle="--",
        alpha=0.6,
    )
    ax1.plot(
        df_plot.index,
        df_plot["MA50"],
        label="50일선",
        color="#3b82f6",
        linestyle="--",
        alpha=0.5,
    )
    ax1.plot(
        df_plot.index,
        df_plot["MA150"],
        label="150일선",
        color="#10b981",
        linewidth=2,
    )
    ax1.plot(
        df_plot.index,
        df_plot["MA200"],
        label="200일선",
        color="#8b5cf6",
        linewidth=1.5,
        alpha=0.5,
    )

    if len(df_hist) >= 200:
        df_hist["MA200_slope"] = df_hist["MA200"].diff(5)
        stage2_df = df_hist[df_hist["MA200_slope"] > 0]

        if len(stage2_df) >= 20:
            bases_info = []

            current_base = 1
            base_start_idx = stage2_df.index[0]
            highest_price = stage2_df["close"].iloc[0]
            in_correction = False

            for idx, row in stage2_df.iterrows():
                price_c = row["close"]
                if price_c > highest_price:
                    if in_correction:
                        bases_info.append(
                            (
                                base_start_idx,
                                idx,
                                highest_price,
                                current_base,
                            )
                        )
                        current_base += 1
                        base_start_idx = idx
                        in_correction = False
                    highest_price = price_c
                elif price_c < highest_price * 0.90:
                    in_correction = True

            bases_info.append(
                (
                    base_start_idx,
                    stage2_df.index[-1],
                    highest_price,
                    current_base,
                )
            )

            for start_dt, end_dt, h_price, b_num in bases_info:
                if (
                    end_dt >= df_plot.index[0]
                    and start_dt <= df_plot.index[-1]
                ):
                    plot_start = max(start_dt, df_plot.index[0])
                    plot_end = min(end_dt, df_plot.index[-1])

                    try:
                        y_start = df_plot.loc[plot_start, "close"]
                        y_end = df_plot.loc[plot_end, "close"]
                        y_max = df_plot.loc[plot_start:plot_end, "close"].max()
                        y_min = df_plot.loc[plot_start:plot_end, "close"].min()
                    except KeyError:
                        continue

                    x_start = mdates.date2num(plot_start)
                    x_end = mdates.date2num(plot_end)
                    x_mid = (x_start + x_end) / 2
                    width = x_end - x_start

                    is_current = b_num == base_stage

                    if is_current:
                        y_control = y_min - (h_price * 0.05)
                        path_data = [
                            (patches.Path.MOVETO, (x_start, y_start)),
                            (patches.Path.CURVE3, (x_mid, y_control)),
                            (patches.Path.CURVE3, (x_end, y_end)),
                        ]
                        codes, verts = zip(*path_data)
                        path = patches.Path(verts, codes)

                        patch = patches.PathPatch(
                            path,
                            edgecolor="#f59e0b",
                            facecolor="none",
                            lw=2,
                            ls="-",
                            alpha=0.9,
                            zorder=4,
                        )
                        ax1.add_patch(patch)

                        ax1.fill_between(
                            df_plot.loc[plot_start:plot_end].index,
                            df_plot.loc[plot_start:plot_end, "close"],
                            y_control,
                            color="#fef3c7",
                            alpha=0.12,
                            zorder=3,
                        )

                        ax1.text(
                            mdates.num2date(x_mid),
                            y_control,
                            f" 현재 Base {b_num}기 ",
                            color="#f59e0b",
                            fontsize=9,
                            fontweight="bold",
                            ha="center",
                            va="top",
                            bbox=dict(
                                boxstyle="round,pad=0.2",
                                facecolor="white",
                                edgecolor="#f59e0b",
                                alpha=0.8,
                                lw=0.5,
                            ),
                        )
                    else:
                        height = (
                            (y_max - y_min)
                            if (y_max > y_min)
                            else (h_price * 0.15)
                        )
                        y_mid = (y_max + y_min) / 2

                        ellipse = patches.Ellipse(
                            xy=(x_mid, y_mid),
                            width=width,
                            height=height,
                            edgecolor="#6366f1",
                            facecolor="#e0e7ff",
                            alpha=0.15,
                            linewidth=1.5,
                            linestyle="--",
                            zorder=3,
                        )
                        ax1.add_patch(ellipse)

                        ellipse_line = patches.Ellipse(
                            xy=(x_mid, y_mid),
                            width=width,
                            height=height,
                            edgecolor="#6366f1",
                            facecolor="none",
                            alpha=0.7,
                            linewidth=1.2,
                            linestyle="--",
                            zorder=4,
                        )
                        ax1.add_patch(ellipse_line)

                        ax1.text(
                            mdates.num2date(x_mid),
                            y_mid - (height / 2),
                            f" 과거 Base {b_num}기 ",
                            color="#6366f1",
                            fontsize=9,
                            fontweight="bold",
                            ha="center",
                            va="top",
                            bbox=dict(
                                boxstyle="round,pad=0.2",
                                facecolor="white",
                                edgecolor="#6366f1",
                                alpha=0.8,
                                lw=0.5,
                            ),
                        )

    info_text = f"▶ 미너비니: {m_point} [{base_stage}기 진행중]\n▶ 와인스타인: {w_point}"
    ax1.text(
        0.02,
        0.92,
        info_text,
        transform=ax1.transAxes,
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#ffffff",
            edgecolor="#cbd5e1",
            alpha=0.9,
        ),
    )

    ax1.set_title(
        f"📈 {name} ({ticker}) 베이스 진행 스크리닝 차트",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax1.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=True,
        facecolor="white",
        edgecolor="#e2e8f0",
        fontsize=9,
    )
    ax1.grid(True, linestyle=":", alpha=0.5)

    colors = [
        "#ef4444" if row["close"] >= row["open"] else "#3b82f6"
        for idx, row in df_plot.iterrows()
    ]
    ax2.bar(
        df_plot.index, df_plot["volume"], color=colors, alpha=0.7, width=0.6
    )
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.set_ylabel("거래량")

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str


def generate_combined_html_report(
    df_result, today_str, total_scanned, passed_count, chart_list
):
    kst = timezone(timedelta(hours=9))
    now_kst = datetime.now(kst).strftime("%Y년 %m월 %d일 %H시 %M분")

    table_rows = ""
    for idx, row in df_result.iterrows():
        rank = idx + 1
        rating = row["추천등급"]
        if rating == "강력매수":
            badge_style = "bg-danger text-white"
        elif rating == "매수":
            badge_style = "bg-primary text-white"
        else:
            badge_style = "bg-warning text-dark"

        vcp_active = (
            "text-success font-bold" if row["변동성축소비율"] < 1.0 else ""
        )
        vol_active = (
            "text-success font-bold" if row["거래량축소비율"] < 1.0 else ""
        )

        table_rows += f"""
        <tr>
            <td class="text-center font-bold" style="font-size: 1.1rem; color: #1e293b;">{rank}위</td>
            <td><span class="ticker-badge">{row['코인코드']}</span></td>
            <td><strong>{row['코인명']}</strong></td>
            <td>{row['현재가']:,}원</td>
            <td class="text-center"><span class="badge {badge_style}" style="padding: 6px 12px; border-radius: 20px; font-weight: bold;">{rating}</span></td>
            <td style="font-size: 0.9rem;">
                <strong>와인스타인:</strong> {row['와인스타인지점']}<br>
                <strong>미너비니:</strong> {row['미너비니지점']} <span class="badge bg-secondary">베이스 {row['미너비니베이스']}기</span>
            </td>
            <td class="text-center {vcp_active}">{row['변동성축소비율']}</td>
            <td class="text-center {vol_active}">{row['거래량축소비율']}</td>
            <td class="text-center text-danger"><strong>{row['최근거래량증가(배)']}배</strong></td>
            <td class="text-center text-primary">{row['150일선이격도']}%</td>
        </tr>
        """

    chart_sections = ""
    for chart in chart_list:
        chart_sections += f"""
        <div class="row align-items-center border-bottom py-4 bg-white px-3 my-3 rounded-3 shadow-sm">
            <div class="col-lg-3">
                <h4 class="fw-bold text-dark mb-1">{chart['rank']}위. {chart['name']}</h4>
                <p class="text-muted small mb-3">{chart['ticker']}</p>
                <div class="p-3 bg-light rounded-3 mb-2" style="font-size: 0.9rem;">
                    <div class="mb-2"><strong>추천 등급:</strong> <span class="badge bg-danger">{chart['rating']}</span></div>
                    <div class="mb-2"><strong>미너비니 단계:</strong> 베이스 {chart['base_stage']}기 현황</div>
                    <div class="mb-2"><strong>150일선 이격:</strong> {chart['disparity']}%</div>
                    <div><strong>종합 스코어:</strong> {chart['score']}점</div>
                </div>
            </div>
            <div class="col-lg-9 text-center">
                <img src="data:image/png;base64,{chart['img_base64']}" class="img-fluid rounded border shadow-xs" alt="차트">
            </div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>업비트 가상화폐 통합 스크리닝 리포트</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ background-color: #f4f6f9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; color: #334155; padding: 20px 0 40px 0; }}
        .card {{ border: none; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); margin-bottom: 30px; }}
        .theory-title {{ border-left: 5px solid #6366f1; padding-left: 12px; font-weight: 700; }}
        .ticker-badge {{ background-color: #f1f5f9; color: #334155; padding: 6px 10px; border-radius: 6px; font-family: monospace; font-weight: bold; }}
        .stat-card {{ background: linear-gradient(135deg, #6366f1, #4f46e5); color: white; border-radius: 16px; padding: 25px; text-align: center; }}
        .table th {{ background-color: #f8fafc; color: #64748b; font-weight: 600; }}
        .font-bold {{ font-weight: bold; }}
        .disclaimer-banner {{ background-color: #fef2f2; border: 1px solid #fecaca; color: #991b1b; padding: 12px 20px; border-radius: 12px; font-size: 0.88rem; text-align: center; line-height: 1.5; margin-bottom: 25px; }}
        .update-time {{ background-color: #e2e8f0; color: #475569; display: inline-block; padding: 6px 16px; border-radius: 20px; font-weight: 600; font-size: 0.92rem; }}
    </style>
</head>
<body>
    <div class="container" style="max-width: 1200px;">
        
        <!-- 최상단 면책 조항 (Disclaimer Banner) -->
        <div class="disclaimer-banner">
            ⚠️ <strong>[주의 및 면책 조항]</strong> 해당 내용은 가상화폐 기술적 조건 검증 및 정보 제공을 위한 <strong>참고용</strong> 자료이며, 투자의 최종 책임은 전적으로 본인에게 있습니다.<br>
            <span style="font-size: 0.82rem; opacity: 0.85;">(Disclaimer: The content provided herein is for informational and educational purposes only. All investment decisions are solely the responsibility of the investor.)</span>
        </div>

        <div class="text-center mb-5">
            <h1 class="fw-extrabold" style="color: #0f172a;">🪙 업비트 미너비니 & 와인스타인 트렌드 리포트</h1>
            <p class="text-muted fs-5 mb-2">분석 기준일: {today_str[:4]}-{today_str[4:6]}-{today_str[6:]} | 가상화폐 주도주 발굴 스크리너</p>
            <div class="update-time mt-1">⏰ 리포트 자동 산출 일시: {now_kst} (KST)</div>
        </div>

        <div class="row mb-4">
            <div class="col-md-4">
                <div class="stat-card">
                    <h5>총 스캔 코인</h5>
                    <h2 class="display-5 fw-bold">{total_scanned}개</h2>
                    <p class="mb-0">업비트 KRW 원화 마켓 전체 전수조사</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="stat-card" style="background: linear-gradient(135deg, #10b981, #059669);">
                    <h5>조건 만족 (통과)</h5>
                    <h2 class="display-5 fw-bold">{passed_count}개</h2>
                    <p class="mb-0">상승 추세 정배열 진입 완료</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="stat-card" style="background: linear-gradient(135deg, #f59e0b, #d97706);">
                    <h5>최종 리포트 등재</h5>
                    <h2 class="display-5 fw-bold">{len(df_result)}개</h2>
                    <p class="mb-0">종합 스코어 최상위 정렬</p>
                </div>
            </div>
        </div>

        <div class="card p-4">
            <h3 class="theory-title mb-3">💡 크립토 매수 타이밍 융합 가이드</h3>
            <div class="row">
                <div class="col-md-6">
                    <div class="p-3 bg-light rounded-3 h-100">
                        <h5 class="fw-bold text-primary">📌 와인스타인 진입 포인트 (장기 이동평균선)</h5>
                        <ul class="mb-0" style="font-size: 0.95rem; line-height: 1.6;">
                            <li><strong>1차 진입 (돌파):</strong> 장기 바닥권 및 150일선을 대량 거래량으로 최초 상방 돌파하는 순간</li>
                            <li><strong>2차 진입 (되돌림):</strong> 첫 돌파 이후 거래량이 급감하며 150일선 근처로 건전하게 조정받을 때</li>
                            <li><strong>3차 진입 (추가 상승):</strong> 지지를 완벽하게 확인한 뒤 다시 전고점을 뚫고 나가는 시점</li>
                        </ul>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="p-3 bg-light rounded-3 h-100">
                        <h5 class="fw-bold text-success">📌 미너비니 진입 포인트 (트렌드 & VCP)</h5>
                        <ul class="mb-0" style="font-size: 0.95rem; line-height: 1.6;">
                            <li><strong>1차 진입 (VCP 완료):</strong> 정배열 주가 범위 내에서 변동성과 거래량이 극도로 감소해 응축된 구간</li>
                            <li><strong>2차 진입 (피벗 돌파):</strong> 수렴을 끝내고 직전 고가를 뚫는 강력한 매수세가 들어오는 지점</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>

        <div class="card p-4">
            <h3 class="theory-title mb-4">🔍 주도 코인 종합 랭킹 리스트</h3>
            <div class="table-responsive">
                <table class="table table-hover align-middle">
                    <thead>
                        <tr class="text-center">
                            <th>순위</th>
                            <th>코인코드</th>
                            <th>코인명</th>
                            <th>현재가</th>
                            <th>추천등급</th>
                            <th>기법별 예상 매수 타점</th>
                            <th>변동성축소</th>
                            <th>거래량축소</th>
                            <th>최근 거래량 증가율</th>
                            <th>150일선 이격도</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="card p-4">
            <h3 class="theory-title mb-4">📊 조건 만족 주도 코인 입체 차트 전체 분석 (미너비니 베이스 추적)</h3>
            <div class="container-fluid px-0">
                {chart_sections}
            </div>
        </div>
    </div>
</body>
</html>
"""
    file_html = f"업비트_추세돌파_리포트_{today_str}.html"
    with open(file_html, "w", encoding="utf-8-sig") as f:
        f.write(html_content)

    # GitHub Pages 서비스를 위한 index.html 메인 파일 복사 생성
    with open("index.html", "w", encoding="utf-8-sig") as f:
        f.write(html_content)

    return file_html


def get_crypto_screener():
    print(
        "🚀 [미너비니 x 와인스타인] 업비트 가상화폐 발굴 엔진을 가동합니다."
    )

    try:
        market_details = pyupbit.get_tickers(fiat="KRW", verbose=True)
        ticker_names = {
            item["market"]: item["korean_name"] for item in market_details
        }
        all_tickers = list(ticker_names.keys())
        total_scanned = len(all_tickers)

    except Exception as e:
        print(
            f"❌ 업비트 시장 데이터를 받아오는 도중 에러가 발생했습니다: {e}"
        )
        return None

    passed_count = 0
    screener_list = []
    history_cache = {}

    print(f"\n⚙️ {total_scanned}개 업비트 원화 코인 분석 시작")

    pbar = tqdm(all_tickers, total=total_scanned)
    for ticker in pbar:
        name = ticker_names.get(ticker, ticker)
        pbar.set_description(f"🔍 [통과: {passed_count}개 | 진행률]")

        try:
            df_hist = pyupbit.get_ohlcv(ticker, interval="day", count=300)
            if df_hist is None or len(df_hist) < 200:
                continue

            df_hist["MA20"] = df_hist["close"].rolling(window=20).mean()
            df_hist["MA50"] = df_hist["close"].rolling(window=50).mean()
            df_hist["MA120"] = df_hist["close"].rolling(window=120).mean()
            df_hist["MA150"] = df_hist["close"].rolling(window=150).mean()
            df_hist["MA200"] = df_hist["close"].rolling(window=200).mean()

            high_52w = (
                df_hist["high"].rolling(window=250, min_periods=1).max().iloc[-1]
            )
            low_52w = (
                df_hist["low"].rolling(window=250, min_periods=1).min().iloc[-1]
            )

            last_row = df_hist.iloc[-1]
            current_close = last_row["close"]

            ma20, ma50, ma120, ma150, ma200 = (
                last_row["MA20"],
                last_row["MA50"],
                last_row["MA120"],
                last_row["MA150"],
                last_row["MA200"],
            )

            if pd.isna([current_close, ma20, ma50, ma120, ma150, ma200]).any():
                continue

            w_cond1 = current_close > ma150
            w_cond2 = ma150 >= df_hist["MA150"].iloc[-20]
            w_cond3 = ma20 > ma150

            m_cond1 = (current_close > ma120) and (current_close > ma200)
            m_cond2 = ma120 > ma200
            m_cond3 = ma20 > df_hist["MA200"].iloc[-20]
            m_cond4 = (ma20 > ma50) and (ma50 > ma120)

            safety_cond = current_close <= (low_52w * 3.5)

            if (w_cond1 and w_cond2 and w_cond3) or (
                m_cond1 and m_cond2 and m_cond4
            ):
                if not safety_cond:
                    continue

                passed_count += 1

                volatility_recent = df_hist["close"].tail(10).std()
                volatility_past = df_hist["close"].iloc[-40:-10].std()
                vcp_ratio = (
                    round(volatility_recent / volatility_past, 2)
                    if volatility_past > 0
                    else 1.0
                )

                volume_recent_shrink = df_hist["volume"].tail(10).mean()
                volume_past_shrink = df_hist["volume"].iloc[-40:-10].mean()
                volume_shrink_ratio = (
                    round(volume_recent_shrink / volume_past_shrink, 2)
                    if volume_past_shrink > 0
                    else 1.0
                )

                recent_volume_spike = df_hist["volume"].tail(3).mean()
                past_volume_spike = df_hist["volume"].iloc[-23:-3].mean()
                volume_spike_ratio = (
                    round(recent_volume_spike / past_volume_spike, 2)
                    if past_volume_spike > 0
                    else 1.0
                )

                disparity_150 = round((current_close / ma150) * 100, 1)
                dist_from_high = round(
                    ((high_52w - current_close) / high_52w) * 100, 1
                )

                if disparity_150 <= 108.0:
                    w_point = "1차 진입 (돌파 초입)"
                elif 108.0 < disparity_150 <= 125.0:
                    if volume_shrink_ratio < 0.9:
                        w_point = "2차 진입 (지선 눌림목)"
                    else:
                        w_point = "3차 진입 (상승 추세 진행)"
                else:
                    w_point = "3차 진입 (추격 매수 주의)"

                if vcp_ratio < 1.0 and volume_shrink_ratio < 1.0:
                    m_point = "1차 진입 (VCP 압축 완료)"
                elif dist_from_high <= 10.0:
                    m_point = "2차 진입 (피벗 돌파 임박)"
                else:
                    m_point = "대기 및 추세 관망"

                base_stage = calculate_minervini_base(df_hist)

                score = 0
                if vcp_ratio < 1.0 and volume_shrink_ratio < 1.0:
                    score += 40
                if volume_spike_ratio >= 1.5:
                    score += 30
                if dist_from_high <= 20.0:
                    score += 20
                if 100.0 <= disparity_150 <= 115.0:
                    score += 10

                if score >= 70:
                    rating = "강력매수"
                elif score >= 40:
                    rating = "매수"
                else:
                    rating = "약한매수"

                tqdm.write(
                    f" 🟢 [조건 충족!] {ticker:<8} ({name}) | 등급: {rating:<5} | 스코어: {score}점 | 베이스: {base_stage}기"
                )

                screener_list.append(
                    {
                        "코인코드": ticker,
                        "코인명": name,
                        "현재가": (
                            float(current_close)
                            if current_close >= 1
                            else round(float(current_close), 4)
                        ),
                        "추천등급": rating,
                        "와인스타인지점": w_point,
                        "미너비니지점": m_point,
                        "미너비니베이스": base_stage,
                        "변동성축소비율": vcp_ratio,
                        "거래량축소비율": volume_shrink_ratio,
                        "최근거래량증가(배)": volume_spike_ratio,
                        "150일선이격도": disparity_150,
                        "종합점수": score,
                    }
                )
                history_cache[ticker] = df_hist

        except Exception:
            continue

    if not screener_list:
        print("\n❌ 오늘 조건 필터를 충족하는 코인이 전혀 없습니다.")
        return None

    df_result = pd.DataFrame(screener_list)
    df_result = df_result.sort_values(
        by=["종합점수", "최근거래량증가(배)"], ascending=[False, False]
    ).reset_index(drop=True)

    df_result_for_charts = df_result.copy()

    df_result.index = df_result.index + 1
    df_result.index.name = "순위"

    today_str = datetime.today().strftime("%Y%m%d")

    file_csv = f"업비트_추세돌파_스크리닝_{today_str}.csv"
    df_result.to_csv(file_csv, index=True, encoding="utf-8-sig")

    print(
        f"\n📊 조건 만족 주도 코인 전체 종목 ({len(df_result_for_charts)}개)에 대한 시각화 차트를 빌드하는 중..."
    )
    chart_list = []

    for idx, row in df_result_for_charts.iterrows():
        ticker = row["코인코드"]
        df_hist_target = history_cache.get(ticker)
        if df_hist_target is not None:
            img_base64 = generate_chart_image(
                ticker=ticker,
                name=row["코인명"],
                df_hist=df_hist_target,
                w_point=row["와인스타인지점"],
                m_point=row["미너비니지점"],
                base_stage=row["미너비니베이스"],
            )
            chart_list.append(
                {
                    "rank": idx + 1,
                    "ticker": ticker,
                    "name": row["코인명"],
                    "rating": row["추천등급"],
                    "base_stage": row["미너비니베이스"],
                    "disparity": row["150일선이격도"],
                    "score": row["종합점수"],
                    "img_base64": img_base64,
                }
            )

    file_html = generate_combined_html_report(
        df_result.reset_index(drop=True),
        today_str,
        total_scanned,
        passed_count,
        chart_list,
    )

    print(
        f"\n🎉 [완료] 조건 만족 전체 코인 ({len(df_result)}개) 스크리닝이 완료되었습니다!"
    )
    print(f"📂 CSV 저장 경로: {file_csv}")
    print(f"🌐 HTML 웹 리포트: {file_html}")
    return df_result


if __name__ == "__main__":
    result = get_crypto_screener()
