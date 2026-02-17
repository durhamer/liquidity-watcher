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
    
    # 抓取足夠長的數據 (30年) 以涵蓋 2000 年
    data_fetch_days = 365 * 30 

    st.markdown("---")
    st.markdown("[申請 FRED API Key](https://fred.stlouisfed.org/docs/api/api_key.html)")

# --- 3. 數據核心 ---
@st.cache_data(ttl=3600)
def get_macro_data(api_key, days):
    fred = Fred(api_key=api_key)
    start_date = datetime.now() - timedelta(days=days)
    
    try:
        # 1. 流動性數據
        fed_assets = fred.get_series('WALCL', observation_start=start_date)
        tga = fred.get_series('WTREGEN', observation_start=start_date)
        # RRP 在 2013 以前不存在，填 0
        rrp = fred.get_series('RRPONTSYD', observation_start=start_date).fillna(0)
        
        # 2. 利率與利差
        yc_10y3m = fred.get_series('T10Y3M', observation_start=start_date)
        t3m = fred.get_series('DGS3MO', observation_start=start_date)
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
        
        # 數據清洗
        df = df.fillna(method='ffill')
        df['RRP'] = df['RRP'].fillna(0)
        df['RRP_Rate'] = df['RRP_Rate'].fillna(0)
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

        # 過濾顯示數據
        display_start_date = f"{display_start_year}-01-01"
        display_df = merged_df[merged_df.index >= display_start_date]

        # --- 側邊欄：原始數據下載 ---
        with st.sidebar:
            st.divider()
            st.subheader("💾 數據匯出")
            csv = display_df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 下載當前圖表數據 (CSV)",
                data=csv,
                file_name=f'macro_data_{display_start_year}_present.csv',
                mime='text/csv',
            )
            st.info("下載後可用 Excel 開啟，驗證數據相關性。")

        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💧 流動性估值", "📉 殖利率曲線", "☢️ VPIN 毒性偵測", "🏦 雙戰場違約監控", "🧮 數學相關性矩陣"
        ])

        # Tab 1: 流動性
        with tab1:
            st.subheader(f"美元淨流動性 vs {compare_index.split(' ')[0]}")
            train_start = f"{reg_start_year}-01-01"
            train_data = merged_df[merged_df.index >= train_start]
            
            if len(train_data) > 30:
                x = train_data['Net_Liquidity']; y = train_data['Stock_Price']
                slope, intercept = np.polyfit(x, y, 1)
                
                merged_df['Fair_Value'] = merged_df['Net_Liquidity'] * slope + intercept
                merged_df['Deviation_Pct'] = ((merged_df['Stock_Price'] - merged_df['Fair_Value']) / merged_df['Fair_Value']) * 100
                
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

        # Tab 4: 雙戰場違約監控
        with tab4:
            st.subheader("🏦 雙戰場違約監控 vs 股價")
            fig_battle = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_battle.add_trace(go.Scatter(
                x=display_df.index, y=display_df['HY_Spread'], 
                name="高收益債恐慌利差 (HY Spread)", 
                fill='tozeroy', 
                line=dict(color='rgba(148, 0, 211, 0.2)', width=0),
                marker=dict(color='rgba(148, 0, 211, 0.2)')
            ), secondary_y=False)

            fig_battle.add_trace(go.Scatter(
                x=display_df.index, y=display_df['Delinq_Consumer'], 
                name="消費者違約率 (Credit Card)", 
                line=dict(color='#FF4500', width=3)
            ), secondary_y=False)
            
            fig_battle.add_trace(go.Scatter(
                x=display_df.index, y=display_df['Delinq_Corp'], 
                name="企業違約率 (C&I Loans)", 
                line=dict(color='#FFD700', width=3, dash='solid')
            ), secondary_y=False)

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
            fig_battle.update_yaxes(title_text="Delinquency / Spread (%)", secondary_y=False)
            fig_battle.update_yaxes(title_text="Stock Price Index", secondary_y=True, showgrid=False)
            st.plotly_chart(fig_battle, use_container_width=True)

        # --- Tab 5: 數學相關性分析 (含下載按鈕) ---
        with tab5:
            st.subheader("🧮 數學真相：相關性矩陣 (Correlation Matrix)")
            st.markdown(f"""
            這裡直接用數據回答你的問題：**「這些風險指標與 {compare_index} 到底有沒有數學相關？」**
            * **數值越接近 1.0 (紅):** 正相關 (同步漲跌)。
            * **數值越接近 -1.0 (藍):** 負相關 (蹺蹺板效應)。
            * **數值接近 0:** 沒關係 (Random)。
            """)
            
            # 準備相關性分析的數據集
            # 我們只選取關鍵指標
            corr_cols = ['Stock_Price', 'Net_Liquidity', 'Delinq_Consumer', 'Delinq_Corp', 'HY_Spread', 'Yield_Curve']
            corr_df = display_df[corr_cols].corr()
            
            # [新增功能] 下載相關性矩陣的 CSV
            csv_corr = corr_df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 下載相關性矩陣數據 (CSV)",
                data=csv_corr,
                file_name=f'correlation_matrix_{display_start_year}_present.csv',
                mime='text/csv',
            )
            
            # 繪製熱力圖
            fig_corr = px.imshow(
                corr_cols_labels := corr_df,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdBu_r', # 紅藍配色 (紅正藍負)
                title=f"Correlation Matrix ({display_start_year}-Present)"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.info("""
            **💡 狙擊手解讀技巧：**
            1. 檢查 **Stock_Price** 與 **Net_Liquidity** 的關係。如果是高度正相關 (紅)，代表這段時間是「資金行情」。
            2. 檢查 **Stock_Price** 與 **HY_Spread**。理論上應該是強烈負相關 (藍)，代表恐慌越低，股價越高。如果變成正相關，代表市場失靈。
            3. 檢查 **Delinq_Consumer** 與 **Delinq_Corp**。看這兩個違約率是否同步。
            """)

else:
    st.info("👈 請在左側輸入 FRED API Key 以啟動交互式戰情室")
