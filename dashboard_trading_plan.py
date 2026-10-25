# -*- coding: utf-8 -*-
"""
Dashboard Trading Plan Saham IDX — Intraday
Sumber harga: Yahoo Finance via yfinance (bukan feed real-time bursa).
Fundamental: prioritas data resmi BEI yang diinput/diunduh pengguna; fallback Yahoo Finance.
Tidak mengirim order dan bukan nasihat investasi.
"""
from datetime import datetime
from pathlib import Path
import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Trading Plan IDX | Intraday", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
.metric-card {border:1px solid rgba(128,128,128,.3); border-radius:12px; padding:14px;}
.small-note {font-size: .88rem; opacity:.8;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Dashboard Trading Plan Saham — Intraday")
st.caption("Bollinger Bands • Stochastic RSI • Chart Pattern • Fundamental • Risk Plan")

with st.sidebar:
    st.header("Pengaturan")
    ticker_input = st.text_input("Ticker saham", value="BBCA", help="Contoh saham IDX: BBCA, BBRI, TLKM, ASII. Tanpa .JK akan ditambahkan otomatis.")
    interval = st.selectbox("Timeframe harga", ["5m", "15m", "30m", "60m", "1d"], index=1)
    period = st.selectbox("Rentang data", ["5d", "1mo", "3mo", "6mo", "1y", "2y"], index=1)
    st.caption("Intraday Yahoo Finance biasanya hanya menyediakan histori intraday terbatas. Rentang yang tersedia tergantung interval.")
    capital = st.number_input("Modal trading (Rp)", min_value=0, value=10_000_000, step=500_000)
    risk_pct = st.number_input("Risiko per transaksi (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    fee_pct = st.number_input("Estimasi total fee beli+jual (%)", min_value=0.0, max_value=5.0, value=0.4, step=0.05)
    st.divider()
    st.subheader("Fundamental resmi BEI")
    fundamentals_file = st.file_uploader(
        "Opsional: unggah CSV/XLSX laporan/rasio resmi",
        type=["csv", "xlsx"],
        help="Unggah data yang sudah Anda unduh dari BEI/emiten. Kolom ideal: metric dan value, atau metrik sebagai kolom pada satu baris."
    )
    st.caption("Tanpa file resmi, dashboard mencoba Yahoo Finance dan memberi label sumber dengan jelas.")

def clean_symbol(raw):
    s = raw.strip().upper().replace(" ", "")
    if not s:
        raise ValueError("Masukkan ticker saham.")
    if "." in s or s.startswith("^"):
        return s
    return s + ".JK"

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)

def indicators(df):
    x = df.copy()
    c = x["Close"]
    x["BB_MID"] = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    x["BB_UPPER"] = x["BB_MID"] + 2*sd
    x["BB_LOWER"] = x["BB_MID"] - 2*sd
    x["RSI"] = rsi(c, 14)
    lo = x["RSI"].rolling(14).min()
    hi = x["RSI"].rolling(14).max()
    raw = 100*(x["RSI"]-lo)/(hi-lo).replace(0, np.nan)
    x["STOCH_K"] = raw.rolling(3).mean()
    x["STOCH_D"] = x["STOCH_K"].rolling(3).mean()
    x["SMA50"] = c.rolling(50).mean()
    x["VOL_MA20"] = x["Volume"].rolling(20).mean()
    prev = x["Close"].shift()
    tr = pd.concat([x["High"]-x["Low"], (x["High"]-prev).abs(), (x["Low"]-prev).abs()], axis=1).max(axis=1)
    x["ATR"] = tr.rolling(14).mean()
    return x

def detect_patterns(df):
    x = df.dropna(subset=["Close"]).tail(100)
    out = []
    if len(x) < 25:
        return ["Data belum cukup untuk deteksi pola (perlu sekitar 25 candle atau lebih)."]
    last = x.iloc[-1]
    prior = x.iloc[-21:-1]
    if len(prior) >= 15:
        if last["Close"] > prior["High"].max():
            out.append("Breakout high 20 candle — cek volume dan retest.")
        if last["Close"] < prior["Low"].min():
            out.append("Breakdown low 20 candle — risiko turun meningkat.")
    high, low = x["High"], x["Low"]
    sh = high[(high.shift(1) < high) & (high.shift(-1) < high)]
    sl = low[(low.shift(1) > low) & (low.shift(-1) > low)]
    if len(sh) >= 2 and abs(sh.iloc[-1]-sh.iloc[-2])/max(abs(sh.iloc[-1]), abs(sh.iloc[-2]), 1e-9) < .025:
        out.append("Potensi Double Top; konfirmasi jika neckline/support ditembus.")
    if len(sl) >= 2 and abs(sl.iloc[-1]-sl.iloc[-2])/max(abs(sl.iloc[-1]), abs(sl.iloc[-2]), 1e-9) < .025:
        out.append("Potensi Double Bottom; konfirmasi jika resistance ditembus.")
    if len(sh) >= 3 and len(sl) >= 3:
        hs, ls = sh.tail(3).values, sl.tail(3).values
        hslope = np.polyfit(np.arange(3), hs, 1)[0]/max(abs(np.mean(hs)),1e-9)
        lslope = np.polyfit(np.arange(3), ls, 1)[0]/max(abs(np.mean(ls)),1e-9)
        if hslope < -.002 and lslope > .002:
            out.append("Potensi Symmetrical Triangle.")
        elif abs(hslope) < .003 and lslope > .002:
            out.append("Potensi Ascending Triangle.")
        elif hslope < -.002 and abs(lslope) < .003:
            out.append("Potensi Descending Triangle.")
        elif hslope > .002 and lslope > .002:
            out.append("Potensi Rising Channel/Wedge; waspadai breakdown.")
        elif hslope < -.002 and lslope < -.002:
            out.append("Potensi Descending Channel.")
    rng = max(float(last["High"]-last["Low"]), 1e-9)
    body = abs(float(last["Close"]-last["Open"]))
    lw = min(float(last["Open"]),float(last["Close"]))-float(last["Low"])
    uw = float(last["High"])-max(float(last["Open"]),float(last["Close"]))
    if body/rng < .12: out.append("Doji / candle ragu-ragu.")
    if lw > 2*max(body,1e-9) and uw < body: out.append("Potensi Hammer; validasi konteks downtrend.")
    if uw > 2*max(body,1e-9) and lw < body: out.append("Potensi Shooting Star; validasi konteks uptrend.")
    return out or ["Belum ada pola utama yang terdeteksi oleh aturan heuristik."]

def safe_float(v):
    try:
        z = float(v)
        return z if np.isfinite(z) else np.nan
    except Exception:
        return np.nan

def load_official_file(file):
    if file is None:
        return {}, None
    try:
        if file.name.lower().endswith(".csv"):
            raw = pd.read_csv(file)
        else:
            raw = pd.read_excel(file)
        if {"metric", "value"}.issubset({str(c).strip().lower() for c in raw.columns}):
            raw.columns = [str(c).strip().lower() for c in raw.columns]
            d = dict(zip(raw["metric"].astype(str).str.strip(), raw["value"]))
        elif len(raw) > 0:
            d = {str(k).strip(): v for k, v in raw.iloc[0].to_dict().items()}
        else:
            return {}, "File tidak berisi baris data."
        return d, None
    except Exception as e:
        return {}, f"File fundamental tidak dapat dibaca: {e}"

def main_analysis(symbol, period, interval):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval, auto_adjust=False, actions=False)
    if hist is None or hist.empty:
        raise RuntimeError("Data harga kosong dari Yahoo Finance. Periksa ticker, interval, dan rentang histori.")
    hist.columns = [str(c).title() for c in hist.columns]
    needed = ["Open","High","Low","Close","Volume"]
    if not all(c in hist.columns for c in needed):
        raise RuntimeError("Kolom OHLCV tidak lengkap pada sumber data.")
    hist = hist[needed].dropna(subset=["Close"])
    return ticker, indicators(hist)

def get_fundamentals(ticker, uploaded):
    official, file_err = load_official_file(uploaded)
    try:
        info = ticker.get_info() or {}
    except Exception:
        info = {}
    mapping = {
        "Nama perusahaan": ("Nama perusahaan", "longName"),
        "Sektor": ("Sektor", "sector"),
        "Industri": ("Industri", "industry"),
        "Market cap": ("Market cap", "marketCap"),
        "Trailing P/E": ("Trailing P/E", "trailingPE"),
        "Forward P/E": ("Forward P/E", "forwardPE"),
        "PBV / Price-to-book": ("PBV / Price-to-book", "priceToBook"),
        "EPS (TTM)": ("EPS (TTM)", "trailingEps"),
        "ROE": ("ROE", "returnOnEquity"),
        "Profit margin": ("Profit margin", "profitMargins"),
        "Operating margin": ("Operating margin", "operatingMargins"),
        "Debt/Equity": ("Debt/Equity", "debtToEquity"),
        "Current ratio": ("Current ratio", "currentRatio"),
        "Revenue growth": ("Revenue growth", "revenueGrowth"),
        "Earnings growth": ("Earnings growth", "earningsGrowth"),
        "Dividend yield": ("Dividend yield", "dividendYield"),
        "52-week high": ("52-week high", "fiftyTwoWeekHigh"),
        "52-week low": ("52-week low", "fiftyTwoWeekLow"),
    }
    out, sources = {}, {}
    norm = {str(k).strip().lower(): v for k,v in official.items()}
    for label, (offkey, yfkey) in mapping.items():
        val = None
        for k,v in norm.items():
            if k == offkey.lower() or k == yfkey.lower() or k.replace("_"," ") == offkey.lower():
                val = v
                break
        if val is not None and pd.notna(val) and str(val).strip() != "":
            out[label], sources[label] = val, "File BEI/emiten yang diunggah"
        elif info.get(yfkey) is not None:
            out[label], sources[label] = info.get(yfkey), "Yahoo Finance (fallback)"
        else:
            out[label], sources[label] = None, "Tidak tersedia"
    return out, sources, file_err

def fundamental_status(f):
    def n(k): return safe_float(f.get(k))
    score, evidence = 0, 0
    roe, margin, debt, current, rev, pe = n("ROE"), n("Profit margin"), n("Debt/Equity"), n("Current ratio"), n("Revenue growth"), n("Trailing P/E")
    if pd.notna(roe):
        evidence += 1
        score += 1 if roe > .15 else (-1 if roe < 0 else 0)
    if pd.notna(margin):
        evidence += 1
        score += 1 if margin > .10 else (-1 if margin < 0 else 0)
    if pd.notna(debt):
        evidence += 1
        score += 1 if debt < 100 else (-1 if debt > 200 else 0)
    if pd.notna(current):
        evidence += 1
        score += 1 if current >= 1.5 else (-1 if current < 1 else 0)
    if pd.notna(rev):
        evidence += 1
        score += 1 if rev > 0 else (-1 if rev < 0 else 0)
    if pd.notna(pe):
        evidence += 1
        score += 1 if 0 < pe < 25 else (-1 if pe > 50 else 0)
    if evidence < 3: return "DATA FUNDAMENTAL TERBATAS", score, evidence
    if score >= 3: return "RELATIF KUAT (indikatif)", score, evidence
    if score <= -2: return "PERLU WASPADA (indikatif)", score, evidence
    return "CAMPURAN (indikatif)", score, evidence

def fmt(v, decimals=2):
    z = safe_float(v)
    return "N/A" if pd.isna(z) else f"{z:,.{decimals}f}"

def rupiah(v):
    z = safe_float(v)
    return "N/A" if pd.isna(z) else f"Rp {z:,.0f}"

if st.button("🔎 Analisis ticker", type="primary", use_container_width=True):
    st.session_state["run_analysis"] = True

if st.session_state.get("run_analysis", False):
    try:
        symbol = clean_symbol(ticker_input)
        ticker, df = main_analysis(symbol, period, interval)
        valid = df.dropna(subset=["BB_MID", "STOCH_K", "STOCH_D"])
        if valid.empty:
            st.error("Data belum cukup untuk menghitung indikator.")
            st.stop()
        last, prev = valid.iloc[-1], valid.iloc[-2]
        price = float(last["Close"])
        atr = safe_float(last["ATR"])
        if pd.isna(atr) or atr <= 0:
            atr = max(float(last["High"]-last["Low"]), price*.005)
        score, reasons = 0, []
        if pd.notna(last["SMA50"]):
            if price > last["SMA50"]: score += 1; reasons.append("harga di atas SMA50")
            else: score -= 1; reasons.append("harga di bawah SMA50")
        if price <= last["BB_LOWER"]: score += 1; reasons.append("harga menyentuh/di bawah lower BB")
        elif price >= last["BB_UPPER"]: score -= 1; reasons.append("harga menyentuh/di atas upper BB")
        k,d = safe_float(last["STOCH_K"]), safe_float(last["STOCH_D"])
        pk,pd_ = safe_float(prev["STOCH_K"]), safe_float(prev["STOCH_D"])
        if all(pd.notna(v) for v in [k,d,pk,pd_]):
            if k < 20 and pk <= pd_ and k > d: score += 2; reasons.append("Stoch RSI bullish cross di oversold")
            elif k > 80 and pk >= pd_ and k < d: score -= 2; reasons.append("Stoch RSI bearish cross di overbought")
            elif k > d: score += 1; reasons.append("%K di atas %D")
            else: score -= 1; reasons.append("%K di bawah %D")
        if score >= 3: tech_status = "BULLISH — tunggu konfirmasi entry"
        elif score <= -2: tech_status = "BEARISH — hindari entry long agresif"
        else: tech_status = "NETRAL / WAIT"
        recent = df.tail(20)
        support, resistance = float(recent["Low"].min()), float(recent["High"].max())
        entry_low, entry_high = max(support, price-.25*atr), min(price+.10*atr, resistance) if resistance >= price else price+.10*atr
        stop = min(support-.25*atr, price-1.2*atr)
        risk = max(price-stop, .01)
        tp1, tp2 = price+1.5*risk, price+2*risk
        risk_budget = capital*risk_pct/100
        qty = int(risk_budget/risk) if risk > 0 else 0
        lot_qty = (qty//100)*100
        f, fsrc, file_err = get_fundamentals(ticker, fundamentals_file)
        fstatus, fscore, fcount = fundamental_status(f)
        if tech_status.startswith("BULLISH") and fstatus.startswith("RELATIF KUAT"):
            overall = "SETUP MENARIK UNTUK DIKAJI — tetap perlu konfirmasi"
        elif tech_status.startswith("BEARISH") or fstatus.startswith("PERLU WASPADA"):
            overall = "WASPADA / TUNGGU"
        else:
            overall = "NETRAL — PERLU KONFIRMASI"

        st.subheader(f"{symbol} — {f.get('Nama perusahaan') or 'Nama perusahaan tidak tersedia'}")
        st.caption(f"Periode: {period} • Interval: {interval} • Candle terakhir: {df.index[-1]} • Sumber harga: Yahoo Finance")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Harga terakhir", rupiah(price))
        c2.metric("Status teknikal", tech_status.split(" — ")[0])
        c3.metric("Stoch RSI %K / %D", f"{fmt(k)} / {fmt(d)}")
        c4.metric("Status fundamental", fstatus)
        st.info(f"**Status gabungan:** {overall}\n\n**Alasan teknikal:** {', '.join(reasons) if reasons else 'Data belum cukup untuk alasan skor.'}")

        left,right = st.columns([1.6,1])
        with left:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Harga"))
            fig.add_trace(go.Scatter(x=df.index,y=df["BB_UPPER"],name="BB Upper",line=dict(width=1)))
            fig.add_trace(go.Scatter(x=df.index,y=df["BB_MID"],name="BB Middle",line=dict(width=1)))
            fig.add_trace(go.Scatter(x=df.index,y=df["BB_LOWER"],name="BB Lower",line=dict(width=1),fill=None))
            fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=25,b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("#### Indikator")
            st.write(f"- Bollinger Upper: **{rupiah(last['BB_UPPER'])}**")
            st.write(f"- Bollinger Middle: **{rupiah(last['BB_MID'])}**")
            st.write(f"- Bollinger Lower: **{rupiah(last['BB_LOWER'])}**")
            st.write(f"- RSI: **{fmt(last['RSI'])}**")
            st.write(f"- SMA50: **{rupiah(last['SMA50'])}**")
            st.write(f"- ATR(14): **{rupiah(atr)}**")
            st.write(f"- Volume: **{fmt(last['Volume'],0)}**")
            st.write(f"- Rata-rata volume 20 candle: **{fmt(last['VOL_MA20'],0)}**")
            st.markdown("#### Stochastic RSI")
            sf = go.Figure()
            sf.add_trace(go.Scatter(x=df.index,y=df["STOCH_K"],name="%K"))
            sf.add_trace(go.Scatter(x=df.index,y=df["STOCH_D"],name="%D"))
            sf.add_hline(y=80,line_dash="dash")
            sf.add_hline(y=20,line_dash="dash")
            sf.update_layout(height=240, yaxis=dict(range=[0,100]), margin=dict(l=0,r=0,t=10,b=0), legend=dict(orientation="h"))
            st.plotly_chart(sf,use_container_width=True)

        st.subheader("Trading plan intraday")
        a,b,c,dcol = st.columns(4)
        a.metric("Area entry pantauan", f"{rupiah(entry_low)} – {rupiah(entry_high)}")
        b.metric("Stop-loss indikatif", rupiah(stop))
        c.metric("Target 1 (1.5R)", rupiah(tp1))
        dcol.metric("Target 2 (2R)", rupiah(tp2))
        st.write(f"Support 20 candle: **{rupiah(support)}** • Resistance 20 candle: **{rupiah(resistance)}**")
        st.write(f"Estimasi jumlah saham berbasis risiko {risk_pct:.1f}%: **{lot_qty:,} saham ({lot_qty//100:,} lot)** sebelum mempertimbangkan fee, slippage, fraksi harga, dan batas likuiditas.")
        if tech_status.startswith("BEARISH"):
            st.warning("Rencana: jangan memaksakan posisi long. Tunggu pembalikan/konfirmasi; angka entry di atas hanya level pantauan.")
        elif tech_status.startswith("NETRAL"):
            st.warning("Rencana: WAIT. Tunggu breakout/retest atau sinyal momentum yang searah.")
        else:
            st.success("Rencana: calon setup long, bukan sinyal beli otomatis. Konfirmasi candle, volume, spread, dan kondisi pasar.")
        st.caption(f"Fee beli+jual yang Anda masukkan: {fee_pct:.2f}%. Fee belum dimodelkan dalam level target/stop-loss; perhitungkan agar target bersih tidak terdistorsi.")

        st.subheader("Deteksi pola chart")
        for p in detect_patterns(df):
            st.write(f"- {p}")
        st.caption("Pola memakai heuristik harga dan swing sederhana; hasil perlu validasi visual. Tidak semua pola klasik dapat diidentifikasi secara andal otomatis.")

        st.subheader("Fundamental")
        if file_err: st.warning(file_err)
        if fundamentals_file is None:
            st.info("Belum ada file fundamental resmi yang diunggah. Data di bawah menggunakan Yahoo Finance jika tersedia; untuk prioritas BEI, unggah CSV/XLSX dari laporan resmi BEI atau emiten.")
        fundamental_rows = []
        percent_keys = {"ROE","Profit margin","Operating margin","Revenue growth","Earnings growth","Dividend yield"}
        for key, val in f.items():
            shown = "Tidak tersedia"
            if val is not None and pd.notna(val):
                if key in percent_keys:
                    z=safe_float(val); shown = f"{z*100:.2f}%" if pd.notna(z) else str(val)
                elif key=="Market cap":
                    shown = f"{safe_float(val):,.0f}" if pd.notna(safe_float(val)) else str(val)
                else:
                    shown = str(val) if isinstance(val,str) else fmt(val)
            fundamental_rows.append({"Metrik":key,"Nilai":shown,"Sumber":fsrc.get(key,"Tidak tersedia")})
        st.dataframe(pd.DataFrame(fundamental_rows), use_container_width=True, hide_index=True)
        st.caption(f"Status fundamental berbasis {fcount} metrik yang terbaca; skor indikatif {fscore}. Bukan valuasi penuh dan ambang sederhana tidak cocok untuk semua sektor, terutama bank/keuangan.")
        st.markdown("**Catatan data resmi:** dashboard tidak mengklaim mengakses API BEI secara langsung. Data BEI/emiten hanya dianggap sebagai sumber prioritas jika Anda mengunggah file terkait; selain itu digunakan fallback Yahoo Finance dengan label sumber per metrik.")
        csv = df.to_csv(index=True).encode("utf-8-sig")
        st.download_button("⬇️ Unduh data indikator CSV", data=csv, file_name=f"indikator_{symbol.replace('.','_')}.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Analisis gagal: {e}")
        st.caption("Periksa koneksi internet, ticker, dan ketersediaan data pada sumber. Yahoo Finance bukan feed bursa real-time yang dijamin.")

st.divider()
st.markdown("""
**Batasan penting**
- Harga berasal dari Yahoo Finance dan dapat tertunda, tidak lengkap, atau berbeda dengan harga broker/BEI. Jangan gunakan untuk eksekusi tanpa verifikasi.
- Untuk intraday, interval dan histori yang tersedia dibatasi oleh provider; data harian bukan pengganti candle intraday.
- File fundamental BEI perlu disiapkan dari sumber resmi dan formatnya mungkin perlu disesuaikan. Kolom yang tidak ditemukan ditampilkan sebagai “Tidak tersedia”, bukan ditebak.
- Stop-loss/target adalah level matematis indikatif; gap, spread, likuiditas, fraksi harga, fee, dan slippage dapat mengubah hasil.
- Aplikasi hanya menganalisis data dan tidak melakukan transaksi.
""")
