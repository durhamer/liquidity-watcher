import streamlit as st
import pandas as pd
from fredapi import Fred
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- 1. 頁面設定 ---
st.set_page_config(page_title="Alpha 宏觀戰情室 Pro (Interactive)", layout="wide")
st.title("🦅 Alpha 宏觀戰情室 Pro (Interactive)")
st.markdown("監控全球資金水位、市場估值與信貸週期的核心儀表板")

# --- 2. 側邊欄：設定 ---
with st.sidebar:
    st.header("⚙️ 參數設定")
    api_key_input = st.text_input("輸入 FRED API Key", type="password")
    st.divider()
    st.subheader("📈 股市對比")
    compare_index = st.selectbox(
        "選擇指數",
        ["^GSPC (S&P 500 - 七巨頭)", "RSP (S&P 500 等權重 - 真實經濟)", "^NDX (Nasdaq 100)", "^SOX (費半)", "BTC-USD (比特幣)"]
    )
    st.subheader("🗓️ 時間軸設定")
    display_start_year = st.slider("圖表顯示起始年", 2000, 2026, 2018)
    st.subheader("🧮 模型訓練區間")
    reg_start_year = st.slider("回歸模型訓練起始年", 2010, 2025, 2020)
    data_fetch_days = 365 * 30 
    st.markdown("---")
    st.markdown("[申請 FRED API Key](https://fred.stlouisfed.org/docs/api/api_key.html)")

# --- 3. 數據核心 ---
@st.cache_data(ttl=3600)
def get_macro_data(api_key, days):
    fred = Fred(api_key=api_key)
    start_date = datetime.now() - timedelta(days=days)
    try:
        # 1. 抓取原始數據
        fed_assets = fred.get_series('WALCL', observation_start=start_date)
        tga = fred.get_series('WTREGEN', observation_start=start_date)
        rrp = fred.get_series('RRPONTSYD', observation_start=start_date).fillna(0)
        yc_10y3m = fred.get_series('T10Y3M', observation_start=start_date)
        t3m = fred.get_series('DGS3MO', observation_start=start_date)
        rrp_rate = fred.get_series('RRPONTSYAWARD', observation_start=start_date).fillna(0)
        hy_spread = fred.get_series('BAMLH0A0HYM2', observation_start=start_date)
        delinq_consumer = fred.get_series('DRCCLACBS', observation_start=start_date)

        df = pd.DataFrame({
            'Fed_Assets': fed_assets, 'TGA': tga, 'RRP': rrp,
            'Yield_Curve': yc_10y3m, 'T3M': t3m, 'RRP_Rate': rrp_rate,
            'HY_Spread': hy_spread, 'Delinq_Consumer': delinq_consumer
        }).fillna(method='ffill').dropna()

        # 🟢 單位校準：統一轉為 Trillions (兆美元)
        df['Net_Liquidity'] = (df['Fed_Assets'] / 1000000) - (df['TGA'] / 1000000) - (df['RRP'] / 1000)
        df['Arb_Spread'] = df['T3M'] - df['RRP_Rate']
        return df
    except Exception as e:
        st.error(f"數據抓取錯誤: {e}"); return None

def get_stock_data(ticker, start_date):
    symbol = ticker.split(" ")[0]
    try:
        df_stock = yf.download(symbol, start=start_date, progress=False)
        if df_stock.empty: return None
        if isinstance(df_stock.columns, pd.MultiIndex): 
            df_stock.columns = df_stock.columns.get_level_values(0)
        stock = df_stock['Close']
        stock.index = stock.index.tz_localize(None)
        return stock
    except: return None

# --- 4. 主邏輯 ---
if api_key_input:
    with st.spinner('正在同步數據...'):
        df_macro = get_macro_data(api_key_input, data_fetch_days)
    
    if df_macro is not None:
        stock_series = get_stock_data(compare_index, df_macro.index[0].strftime('%Y-%m-%d'))
        if stock_series is not None:
            merged_df = pd.concat([df_macro, stock_series], axis=1).dropna()
            merged_df.columns = list(df_macro.columns) + ['Stock_Price']
            
            # --- Tab 1: 流動性估值 ---
            tab1, tab2, tab3, tab4 = st.tabs(["💧 流動性估值", "📉 殖利率曲線", "☢️ VPIN", "🏦 違約監控"])

            with tab1:
                st.subheader(f"美元淨流動性 vs {compare_index.split(' ')[0]}")
                
                # 🟢 數據平滑化與模型計算
                merged_df['Net_Liquidity_Smooth'] = merged_df['Net_Liquidity'].rolling(window=7).mean()
                train_data = merged_df[merged_df.index >= f"{reg_start_year}-01-01"].dropna()
                
                if len(train_data) > 30:
                    x = train_data['Net_Liquidity_Smooth']
                    y = train_data['Stock_Price']
                    slope, intercept = np.polyfit(x, y, 1)
                    r_squared = np.corrcoef(x, y)[0,1]**2
                    
                    merged_df['Fair_Value'] = merged_df['Net_Liquidity_Smooth'] * slope + intercept
                    merged_df['Deviation_Pct'] = ((merged_df['Stock_Price'] - merged_df['Fair_Value']) / merged_df['Fair_Value']) * 100
                    
                    latest = merged_df.iloc[-1]
                    plot_df = merged_df[merged_df.index >= f"{display_start_year}-01-01"]

                    # 1. 頂部核心指標
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("淨流動性", f"${latest['Net_Liquidity']:.2f} T")
                    c2.metric("公允股價", f"{latest['Fair_Value']:.0f}")
                    c3.metric("溢價率", f"{latest['Deviation_Pct']:.1f}%", delta_color="inverse")
                    c4.metric("模型解釋力 R²", f"{r_squared:.1%}")

                    # 2. 🟢 新增：單獨的流動性組成拆解圖
                    st.markdown("#### 🌊 流動性組成拆解 (Fed Assets - TGA - RRP)")
                    fig_liq = go.Figure()
                    fig_liq.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Net_Liquidity'], name="Net Liquidity", line=dict(color='#00FF00', width=3)))
                    fig_liq.add_trace(go.Scatter(x=plot_df.index, y=plot_df['TGA']/1000000, name="TGA (Freezer)", line=dict(color='#FF4136', width=1, dash='dot')))
                    fig_liq.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RRP']/1000, name="RRP (Water Tank)", line=dict(color='#FFA500', width=1, dash='dash')))
                    fig_liq.update_layout(height=350, template="plotly_dark", margin=dict(t=20, b=20), hovermode="x unified")
                    st.plotly_chart(fig_liq, use_container_width=True)

                    st.divider()

                    # 3. 股價 vs 公允價值圖
                    st.markdown(f"#### ⚖️ 市場估值偏差分析 (Training Start: {reg_start_year})")
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Stock_Price'], name="Price", line=dict(color='#FFA500')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Fair_Value'], name="Fair Value", line=dict(color='#1E90FF', dash='dash')), row=1, col=1)
                    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Deviation_Pct'], name="Bubble %", marker_color=np.where(plot_df['Deviation_Pct']>0, 'red', 'green')), row=2, col=1)
                    fig.update_layout(height=600, template="plotly_dark", hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("雙重利差監控")
                fig_yc = go.Figure()
                fig_yc.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Yield_Curve'], name="10Y-3M (Macro)", line=dict(color='#00FFFF')))
                fig_yc.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Arb_Spread'], name="3M-RRP (Micro)", line=dict(color='#FF00FF', dash='dot')))
                fig_yc.update_layout(height=500, template="plotly_dark"); st.plotly_chart(fig_yc, use_container_width=True)

            with tab4:
                st.subheader("🏦 信貸違約雙戰場")
                fig_battle = make_subplots(specs=[[{"secondary_y": True}]])
                fig_battle.add_trace(go.Scatter(x=plot_df.index, y=plot_df['HY_Spread'], name="HY Spread", fill='tozeroy', line=dict(color='rgba(148, 0, 211, 0.2)', width=0)), secondary_y=False)
                fig_battle.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Delinq_Consumer'], name="Consumer Delinq", line=dict(color='#FF4500', width=3)), secondary_y=False)
                fig_battle.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Stock_Price'], name="Price", line=dict(color='#00FF7F', width=2, dash='dot')), secondary_y=True)
                fig_battle.update_layout(height=600, template="plotly_dark"); st.plotly_chart(fig_battle, use_container_width=True)
else:
    st.info("👈 請在左側輸入 FRED API Key 以啟動")
