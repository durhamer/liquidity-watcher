import streamlit as st
import pandas as pd
from fredapi import Fred
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
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
    # [新增] 圖表顯示起始年
    display_start_year = st.slider("圖表顯示起始年", 2000, 2026, 2020)
    
    st.subheader("🧮 模型訓練區間")
    reg_start_year = st.slider("回歸模型訓練起始年", 2010, 2025, 2020)
    
    # 計算需要的數據回溯天數 (為了模型訓練，可能需要比顯示更早的數據)
    # 這裡預設抓取足夠長的數據，顯示時再過濾
    data_fetch_days = 365 * 20 # 抓 20 年數據確保足夠

    st.markdown("---")
    st.markdown("[申請 FRED API Key](https://fred.stlouisfed.org/docs/api/api_key.html)")

# --- 修正 1. 側邊欄設定區 (Side Bar) ---
# 把原本的 data_fetch_days 計算改成這樣：
# 為了確保能涵蓋到 2000 年 (甚至更早)，我們直接抓 30 年的數據
data_fetch_days = 365 * 30 


# --- 修正 2. 數據抓取核心函數 (Function) ---
@st.cache_data(ttl=3600)
def get_macro_data(api_key, days):
    fred = Fred(api_key=api_key)
    start_date = datetime.now() - timedelta(days=days)
    
    try:
        # 1. 流動性數據
        fed_assets = fred.get_series('WALCL', observation_start=start_date)
        tga = fred.get_series('WTREGEN', observation_start=start_date)
        
        # [修正重點] RRP 在 2013 以前不存在，抓下來會是 NaN。
        # 我們用 fillna(0) 把空值填為 0，這樣就不會導致前面的數據被 dropna 殺掉。
        rrp = fred.get_series('RRPONTSYD', observation_start=start_date).fillna(0)
        
        # 2. 利率與利差
        yc_10y3m = fred.get_series('T10Y3M', observation_start=start_date)
        t3m = fred.get_series('DGS3MO', observation_start=start_date)
        
        # [修正重點] RRP 利率以前也是 0
        rrp_rate = fred.get_series('RRPONTSYAWARD', observation_start=start_date).fillna(0)

        # 3. 信貸週期數據
        bank_credit = fred.get_series('TOTBKCR', observation_start=start_date)
        delinq_consumer = fred.get_series('DRCCLACBS', observation_start=start_date)
        delinq_corp = fred.get_series('DRBLACBS', observation_start=start_date)
        hy_spread = fred.get_series('BAMLH0A0HYM2', observation_start=start_date)

        df = pd.DataFrame({
            'Fed_Assets': fed_assets, 'TGA': tga, 'RRP': rrp,
            'Yield_Curve': yc_10y3m, 
            'T3M': t3m, 'RRP_Rate': rrp_rate,
            'Bank_Credit': bank_credit, 
            'Delinq_Consumer': delinq_consumer,
            'Delinq_Corp': delinq_corp,
            'HY_Spread': hy_spread
        })
        
        # 先做 forward fill 補齊週末或國定假日的空缺
        df = df.fillna(method='ffill')
        
        # [修正重點] 再次確保 RRP 系列是 0 而不是 NaN (雙重保險)
        df['RRP'] = df['RRP'].fillna(0)
        df['RRP_Rate'] = df['RRP_Rate'].fillna(0)
        
        # 最後才 dropna，這時候只會刪除那些「真的完全沒數據」的早期日子 (例如 80 年代)
        df = df.dropna()
        
        # 計算衍生指標
        df['Net_Liquidity'] = (df['Fed_Assets'] - df['TGA'] - df['RRP']) / 1000000 
        df['Arb_Spread'] = df['T3M'] - df['RRP_Rate']
        
        return df
    except Exception as e:
        st.error(f"數據抓取錯誤: {e}")
        return None

def get_stock_data(ticker, start_date):
    if not ticker: return None
    symbol = ticker.split(" ")[0]
    try:
        stock = yf.download(symbol, start=start_date, progress=False)['Close']
        if isinstance(stock, pd.DataFrame): 
             stock = stock.iloc[:, 0]
        stock.index = stock.index.tz_localize(None)
        return stock
    except:
        return None

# --- VPIN 引擎 ---
def calculate_vpin(df, bucket_volume, window=50):
    df = df.copy()
    df['dP'] = df['Close'].diff()
    sigma = df['dP'].std()
    if sigma == 0: sigma = 0.0001
    prob_buy = norm.cdf(df['dP'] / sigma)
    df['Buy_Vol'] = df['Volume'] * prob_buy
    df['Sell_Vol'] = df['Volume'] * (1 - prob_buy)
    df['Cum_Vol'] = df['Volume'].cumsum()
    df['Bucket_ID'] = (df['Cum_Vol'] // bucket_volume).astype(int)
    buckets = df.groupby('Bucket_ID').agg({'Buy_Vol': 'sum', 'Sell_Vol': 'sum', 'Close': 'last', 'Datetime': 'last'})
    buckets['OI'] = (buckets['Buy_Vol'] - buckets['Sell_Vol']).abs()
    buckets['VPIN'] = buckets['OI'].rolling(window=window).sum() / (bucket_volume * window)
    return buckets

# --- 4. 主邏輯 ---
if api_key_input:
    with st.spinner('正在初始化量子數據鏈接...'):
        df = get_macro_data(api_key_input, data_fetch_days)
        
    if df is not None:
        stock_series = get_stock_data(compare_index, df.index[0].strftime('%Y-%m-%d'))
        merged_df = pd.concat([df, stock_series], axis=1).dropna()
        merged_df.columns = list(df.columns) + ['Stock_Price']

        # [關鍵] 根據使用者選擇的年份過濾顯示數據
        display_start_date = f"{display_start_year}-01-01"
        display_df = merged_df[merged_df.index >= display_start_date]

        tab1, tab2, tab3, tab4 = st.tabs([
            "💧 流動性估值", "📉 殖利率曲線", "☢️ VPIN 毒性偵測", "🏦 雙戰場違約監控"
        ])

        # Tab 1: 流動性
        with tab1:
            st.subheader(f"美元淨流動性 vs {compare_index.split(' ')[0]}")
            
            # 模型訓練區間 (可以跟顯示區間不同)
            train_start = f"{reg_start_year}-01-01"
            train_data = merged_df[merged_df.index >= train_start]
            
            if len(train_data) > 30:
                x = train_data['Net_Liquidity']; y = train_data['Stock_Price']
                slope, intercept = np.polyfit(x, y, 1)
                
                # 計算全區間的 Fair Value
                merged_df['Fair_Value'] = merged_df['Net_Liquidity'] * slope + intercept
                merged_df['Deviation_Pct'] = ((merged_df['Stock_Price'] - merged_df['Fair_Value']) / merged_df['Fair_Value']) * 100
                
                # 再切分顯示用的 DF
                plot_df = merged_df[merged_df.index >= display_start_date]
                latest = plot_df.iloc[-1]
                
                c1, c2, c3 = st.columns(3)
                c1.metric("當前淨流動性", f"${latest['Net_Liquidity']:.2f} T")
                c2.metric("理論公允股價", f"{latest['Fair_Value']:.0f}")
                c3.metric("溢價率", f"{latest['Deviation_Pct']:.1f}%", delta_color="inverse")
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Stock_Price'], name="Price", line=dict(color='#FFA500')), row=1, col=1)
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Fair_Value'], name="Fair Value", line=dict(color='#1E90FF', dash='dash')), row=1, col=1)
                fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Deviation_Pct'], name="Bubble %", marker_color=np.where(plot_df['Deviation_Pct']>0, 'red', 'green')), row=2, col=1)
                fig.update_layout(height=700, hovermode="x unified"); st.plotly_chart(fig, use_container_width=True)

        # Tab 2: 殖利率
        with tab2:
            st.subheader("雙重利差監控")
            fig_yc = go.Figure()
            fig_yc.add_trace(go.Scatter(x=display_df.index, y=display_df['Yield_Curve'], name="10Y-3M (Macro)", line=dict(color='#00FFFF')))
            fig_yc.add_trace(go.Scatter(x=display_df.index, y=display_df['Arb_Spread'], name="3M-RRP (Micro)", line=dict(color='#FF00FF', dash='dot')))
            fig_yc.add_hrect(y0=0, y1=-2, fillcolor="red", opacity=0.15, line_width=0)
            fig_yc.update_layout(height=600, hovermode="x unified"); st.plotly_chart(fig_yc, use_container_width=True)

        # Tab 3: VPIN
        with tab3:
            st.subheader("☢️ VPIN 訂單流毒性偵測")
            ticker_map = {"^GSPC": "SPY", "RSP": "RSP", "^NDX": "QQQ", "^SOX": "SOXX", "BTC-USD": "BTC-USD"}
            vpin_symbol = ticker_map.get(compare_index.split(' ')[0], compare_index.split(' ')[0])
            if st.button("🚀 啟動 VPIN 掃描", type="primary"):
                with st.spinner("正在計算..."):
                    try:
                        df_1m = yf.download(vpin_symbol, period='5d', interval='1m', progress=False)
                        if len(df_1m) > 0:
                            if isinstance(df_1m.columns, pd.MultiIndex): df_1m.columns = df_1m.columns.get_level_values(0)
                            df_1m = df_1m.reset_index()
                            if 'Datetime' not in df_1m.columns: df_1m.rename(columns={'index': 'Datetime'}, inplace=True)
                            vpin_data = calculate_vpin(df_1m, bucket_volume=int(df_1m['Volume'].mean()*15))
                            fig_vpin = go.Figure()
                            fig_vpin.add_trace(go.Scatter(x=vpin_data['Datetime'], y=vpin_data['VPIN'], name="VPIN", line=dict(color='#00FF00')))
                            fig_vpin.add_hline(y=0.8, line_color="red"); fig_vpin.update_layout(height=500); st.plotly_chart(fig_vpin, use_container_width=True)
                    except: st.error("數據下載失敗")

        # Tab 4: 雙戰場違約監控 (已更新)
        with tab4:
            st.subheader("🏦 雙戰場違約監控 vs 股價")
            st.markdown("""
            此圖表展示了著名的 **「鱷魚口 (Alligator Jaws)」** 現象：
            * **左軸 (風險指標):** 消費者違約率 (紅)、企業違約率 (黃)、高收益債利差 (紫)。這些越高越危險。
            * **右軸 (資產價格):** **S&P 500 股價 (綠虛線)**。
            * **戰術訊號:** 當 **紅/黃/紫線往上飆**，但 **綠線 (股價) 還在往上** 時，就是最危險的背離。鱷魚嘴巴張得越大，閉合時咬得越用力。
            """)
            
            # 使用 display_df (根據年份過濾後的數據)
            fig_battle = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 1. 企業恐慌 (背景) - 左軸
            fig_battle.add_trace(go.Scatter(
                x=display_df.index, y=display_df['HY_Spread'], 
                name="高收益債恐慌利差 (HY Spread)", 
                fill='tozeroy', 
                line=dict(color='rgba(148, 0, 211, 0.2)', width=0),
                marker=dict(color='rgba(148, 0, 211, 0.2)')
            ), secondary_y=False) # 改為左軸，統一風險指標

            # 2. 消費者違約 (紅線) - 左軸
            fig_battle.add_trace(go.Scatter(
                x=display_df.index, y=display_df['Delinq_Consumer'], 
                name="消費者違約率 (Credit Card)", 
                line=dict(color='#FF4500', width=3)
            ), secondary_y=False)
            
            # 3. 企業違約 (黃線) - 左軸
            fig_battle.add_trace(go.Scatter(
                x=display_df.index, y=display_df['Delinq_Corp'], 
                name="企業違約率 (C&I Loans)", 
                line=dict(color='#FFD700', width=3, dash='solid')
            ), secondary_y=False)

            # 4. [新增] 股價 (綠虛線) - 右軸
            fig_battle.add_trace(go.Scatter(
                x=display_df.index, y=display_df['Stock_Price'],
                name=f"{compare_index.split(' ')[0]} Price",
                line=dict(color='#00FF7F', width=2, dash='dot')
            ), secondary_y=True)

            fig_battle.update_layout(
                height=650, 
                title_text="Risk Metrics vs Asset Price (The Alligator Jaws)",
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1)
            )
            
            # 設定座標軸標題
            fig_battle.update_yaxes(title_text="Delinquency / Spread (%)", secondary_y=False)
            fig_battle.update_yaxes(title_text="Stock Price Index", secondary_y=True, showgrid=False) # 右軸不顯示網格，避免混亂
            
            st.plotly_chart(fig_battle, use_container_width=True)
            
            latest_cons = display_df['Delinq_Consumer'].iloc[-1]
            latest_spread = display_df['HY_Spread'].iloc[-1]
            latest_price = display_df['Stock_Price'].iloc[-1]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 消費者違約率", f"{latest_cons:.2f}%", delta_color="inverse")
            c2.metric("🟣 企業恐慌利差", f"{latest_spread:.2f}%", delta_color="inverse")
            c3.metric("🟢 股價指數", f"{latest_price:,.0f}")

else:
    st.info("👈 請在左側輸入 FRED API Key 以啟動交互式戰情室")
