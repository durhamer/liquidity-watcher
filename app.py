import streamlit as st
import pandas as pd
from fredapi import Fred
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm  # 用於 VPIN 計算

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
    
    st.subheader("🧮 模型訓練區間")
    reg_start_year = st.slider("回歸起始年", 2018, 2024, 2020)
    
    days_back = st.slider("顯示回溯天數", min_value=365, max_value=3650, value=1095, step=30)
    
    st.markdown("---")
    st.markdown("[申請 FRED API Key](https://fred.stlouisfed.org/docs/api/api_key.html)")

# --- 3. 數據核心 ---
@st.cache_data(ttl=3600)
def get_macro_data(api_key, days):
    fred = Fred(api_key=api_key)
    start_date = datetime.now() - timedelta(days=days)
    
    try:
        # 1. 既有流動性數據
        fed_assets = fred.get_series('WALCL', observation_start=start_date)
        tga = fred.get_series('WTREGEN', observation_start=start_date)
        rrp = fred.get_series('RRPONTSYD', observation_start=start_date)
        yc_10y3m = fred.get_series('T10Y3M', observation_start=start_date)
        ccc = fred.get_series('BAMLH0A3HYC', observation_start=start_date)
        bb = fred.get_series('BAMLH0A1HYBB', observation_start=start_date)
        
        # 2. RRP套利利差
        t3m = fred.get_series('DGS3MO', observation_start=start_date)
        rrp_rate = fred.get_series('RRPONTSYAWARD', observation_start=start_date)

        # 3. [新增] 銀行信貸與違約指標
        # TOTBKCR: 美國商業銀行總信貸 (Oxygen)
        # DRCCLACBS: 信用卡貸款違約率 (Poison - 季度數據，需 ffill)
        bank_credit = fred.get_series('TOTBKCR', observation_start=start_date)
        delinquency = fred.get_series('DRCCLACBS', observation_start=start_date)

        df = pd.DataFrame({
            'Fed_Assets': fed_assets, 'TGA': tga, 'RRP': rrp,
            'Yield_Curve': yc_10y3m, 'CCC': ccc, 'BB': bb,
            'T3M': t3m, 'RRP_Rate': rrp_rate,
            'Bank_Credit': bank_credit, 'Delinquency': delinquency
        })
        
        # 處理頻率不一致 (違約率是季度，信貸是週度)
        df = df.fillna(method='ffill').dropna()
        
        # 計算衍生指標
        df['Net_Liquidity'] = (df['Fed_Assets'] - df['TGA'] - df['RRP']) / 1000000 
        df['Credit_Stress'] = df['CCC'] - df['BB']
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
        if isinstance(stock, pd.DataFrame): # Handle yfinance update
             stock = stock.iloc[:, 0]
        stock.index = stock.index.tz_localize(None)
        return stock
    except:
        return None

# --- 新增：VPIN 計算引擎 ---
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
    
    buckets = df.groupby('Bucket_ID').agg({
        'Buy_Vol': 'sum',
        'Sell_Vol': 'sum',
        'Close': 'last',
        'Datetime': 'last'
    })
    
    buckets['OI'] = (buckets['Buy_Vol'] - buckets['Sell_Vol']).abs()
    buckets['VPIN'] = buckets['OI'].rolling(window=window).sum() / (bucket_volume * window)
    
    return buckets

# --- 4. 主邏輯 ---
if api_key_input:
    with st.spinner('正在初始化量子數據鏈接...'):
        df = get_macro_data(api_key_input, days_back + 365)
        
    if df is not None:
        stock_series = get_stock_data(compare_index, df.index[0].strftime('%Y-%m-%d'))
        merged_df = pd.concat([df, stock_series], axis=1).dropna()
        merged_df.columns = list(df.columns) + ['Stock_Price']

        # [更新] 增加第五個 Tab
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💧 流動性估值", "📉 殖利率曲線", "🔥 信用利差", "☢️ VPIN 毒性偵測", "🏦 銀行與違約"
        ])

        with tab1:
            st.subheader(f"美元淨流動性 vs {compare_index.split(' ')[0]}")
            
            train_start = f"{reg_start_year}-01-01"
            train_data = merged_df[merged_df.index >= train_start]
            
            if len(train_data) > 30:
                x = train_data['Net_Liquidity']
                y = train_data['Stock_Price']
                slope, intercept = np.polyfit(x, y, 1)
                
                correlation_matrix = np.corrcoef(x, y)
                correlation_xy = correlation_matrix[0, 1]
                r_squared = correlation_xy ** 2
                
                merged_df['Fair_Value'] = merged_df['Net_Liquidity'] * slope + intercept
                merged_df['Deviation'] = merged_df['Stock_Price'] - merged_df['Fair_Value']
                merged_df['Deviation_Pct'] = (merged_df['Deviation'] / merged_df['Fair_Value']) * 100
                
                latest = merged_df.iloc[-1]

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("當前淨流動性", f"${latest['Net_Liquidity']:.2f} T")
                c2.metric("理論公允股價", f"{latest['Fair_Value']:.0f}")
                is_bubble = latest['Deviation_Pct'] > 0
                c3.metric("⚠️ 溢價率" if is_bubble else "✅ 折價率", f"{latest['Deviation_Pct']:.1f}%", delta_color="inverse")
                c4.metric("模型可信度 (R²)", f"{r_squared:.2f}", delta_color="normal" if r_squared > 0.7 else "inverse")

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.03, row_heights=[0.7, 0.3],
                                    subplot_titles=(f"Price vs Liquidity Model ({reg_start_year}-Present)", "Deviation % (Bubble/Discount)"))

                fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['Stock_Price'], name="Actual Price", line=dict(color='#FFA500', width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['Fair_Value'], name="Fair Value", line=dict(color='#1E90FF', width=2, dash='dash')), row=1, col=1)

                fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['Deviation_Pct'], name="Deviation %", 
                                         fill='tozeroy', line=dict(color='gray', width=0.5),
                                         fillcolor='rgba(200, 200, 200, 0.2)'), row=2, col=1)
                colors = np.where(merged_df['Deviation_Pct'] > 0, 'rgba(255, 0, 0, 0.5)', 'rgba(0, 255, 0, 0.5)')
                fig.add_trace(go.Bar(x=merged_df.index, y=merged_df['Deviation_Pct'], name="Bubble/Crash", marker_color=colors), row=2, col=1)

                fig.update_layout(height=700, hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", y=1.1), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("數據不足，無法計算模型。")

        with tab2:
            st.subheader("雙重利差監控")
            fig_yc = go.Figure()
            fig_yc.add_trace(go.Scatter(x=df.index, y=df['Yield_Curve'], name="10Y-3M (Macro)", line=dict(color='#00FFFF', width=2)))
            fig_yc.add_trace(go.Scatter(x=df.index, y=df['Arb_Spread'], name="3M T-Bill - RRP (Micro)", line=dict(color='#FF00FF', width=2, dash='dot')))
            fig_yc.add_hrect(y0=0, y1=min(df['Yield_Curve'].min(), -1.0), fillcolor="red", opacity=0.15, line_width=0, annotation_text="Recession Zone", annotation_position="bottom right")
            fig_yc.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.8)
            fig_yc.update_layout(height=600, hovermode="x unified", legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig_yc, use_container_width=True)

        with tab3:
            st.subheader("垃圾債壓力指標 (CCC - BB)")
            fig_cs = go.Figure()
            fig_cs.add_trace(go.Scatter(x=df.index, y=df['Credit_Stress'], name="Credit Stress", fill='tozeroy', line=dict(color='firebrick')))
            fig_cs.update_layout(hovermode="x unified")
            st.plotly_chart(fig_cs, use_container_width=True)

        with tab4:
            st.subheader("☢️ VPIN 訂單流毒性偵測 (微觀結構)")
            st.markdown("當 VPIN > 0.8 時，代表市場極度不穩定 (Crash Risk)。")
            
            ticker_map = {"^GSPC": "SPY", "RSP": "RSP", "^NDX": "QQQ", "^SOX": "SOXX", "BTC-USD": "BTC-USD"}
            raw_symbol = compare_index.split(' ')[0]
            vpin_symbol = ticker_map.get(raw_symbol, raw_symbol)

            st.write(f"正在分析標的： **{vpin_symbol}**")
            
            if st.button("🚀 啟動 VPIN 掃描", type="primary"):
                with st.spinner("正在計算流體力學..."):
                    try:
                        df_1m = yf.download(vpin_symbol, period='5d', interval='1m', progress=False)
                        if len(df_1m) > 0:
                            if isinstance(df_1m.columns, pd.MultiIndex):
                                df_1m.columns = df_1m.columns.get_level_values(0)
                            df_1m = df_1m.reset_index()
                            if 'Datetime' not in df_1m.columns: df_1m.rename(columns={'index': 'Datetime'}, inplace=True)
                            
                            avg_vol = df_1m['Volume'].mean()
                            dynamic_bucket = int(avg_vol * 15) 
                            
                            vpin_data = calculate_vpin(df_1m, bucket_volume=dynamic_bucket)
                            
                            fig_vpin = go.Figure()
                            fig_vpin.add_trace(go.Scatter(x=vpin_data['Datetime'], y=vpin_data['VPIN'], name="VPIN Index", line=dict(color='#00FF00', width=2)))
                            fig_vpin.add_hline(y=0.6, line_dash="dash", line_color="orange")
                            fig_vpin.add_hline(y=0.8, line_dash="solid", line_color="red")
                            fig_vpin.add_hrect(y0=0.8, y1=1.0, fillcolor="red", opacity=0.2, line_width=0)
                            
                            fig_vpin.update_layout(height=500, title=f"VPIN Toxicity: {vpin_symbol}", yaxis_title="VPIN (0-1)", hovermode="x unified", yaxis_range=[0, 1.0])
                            st.plotly_chart(fig_vpin, use_container_width=True)
                            
                            latest_vpin = vpin_data['VPIN'].iloc[-1]
                            if latest_vpin > 0.8: st.error(f"🚨 嚴重警告：VPIN = {latest_vpin:.2f}。市場毒性極高！")
                            elif latest_vpin > 0.6: st.warning(f"⚠️ 注意：VPIN = {latest_vpin:.2f}。流動性變薄。")
                            else: st.success(f"✅ 安全：VPIN = {latest_vpin:.2f}。")
                        else:
                            st.error("無法下載數據。")
                    except Exception as e:
                        st.error(f"錯誤: {e}")

        # [新增] Tab 5: 銀行信貸與違約
        with tab5:
            st.subheader("🏦 信貸週期監控：氧氣 vs 毒藥")
            st.markdown("""
            此圖表監控實體經濟的真實健康狀況：
            * **藍色區域 (左軸):** **銀行總信貸 (TOTBKCR)**。這是經濟的「氧氣」。如果曲線轉折向下，代表銀行正在「縮表」，通常是嚴重衰退的前兆。
            * **紅色線條 (右軸):** **信用卡違約率 (Delinquency Rate)**。這是經濟的「毒藥」。當此數值突破 3% 且加速上升時，代表底層消費者的現金流斷裂。
            """)
            
            # 建立雙軸圖表
            fig_bank = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 銀行信貸 (氧氣)
            fig_bank.add_trace(go.Scatter(
                x=df.index, y=df['Bank_Credit'], 
                name="銀行總信貸 (Billions $)", 
                fill='tozeroy', 
                line=dict(color='rgba(30, 144, 255, 0.5)', width=1)
            ), secondary_y=False)
            
            # 違約率 (毒藥)
            fig_bank.add_trace(go.Scatter(
                x=df.index, y=df['Delinquency'], 
                name="信用卡違約率 (%)", 
                line=dict(color='red', width=3)
            ), secondary_y=True)
            
            fig_bank.update_layout(
                height=600, 
                title_text="Bank Credit Cycle vs Consumer Stress",
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1)
            )
            
            fig_bank.update_yaxes(title_text="Total Bank Credit ($B)", secondary_y=False)
            fig_bank.update_yaxes(title_text="Delinquency Rate (%)", secondary_y=True)
            
            st.plotly_chart(fig_bank, use_container_width=True)
            
            # 簡易判讀
            latest_credit = df['Bank_Credit'].iloc[-1]
            latest_delinq = df['Delinquency'].iloc[-1]
            
            c1, c2 = st.columns(2)
            c1.metric("當前銀行信貸水位", f"${latest_credit:,.0f} B")
            c2.metric("當前違約率", f"{latest_delinq:.2f}%", delta_color="inverse")

else:
    st.info("👈 請在左側輸入 FRED API Key 以啟動交互式戰情室")
