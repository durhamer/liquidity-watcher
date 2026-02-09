import streamlit as st
import pandas as pd
from fredapi import Fred
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm  # 新增：用於 VPIN 計算

# --- 1. 頁面設定 ---
st.set_page_config(page_title="Alpha 宏觀戰情室 Pro (Interactive)", layout="wide")
st.title("🦅 Alpha 宏觀戰情室 Pro (Interactive)")
st.markdown("監控全球資金水位與市場估值的核心儀表板")

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
        # 1. 既有數據
        fed_assets = fred.get_series('WALCL', observation_start=start_date)
        tga = fred.get_series('WTREGEN', observation_start=start_date)
        rrp = fred.get_series('RRPONTSYD', observation_start=start_date)
        yc_10y3m = fred.get_series('T10Y3M', observation_start=start_date)
        ccc = fred.get_series('BAMLH0A3HYC', observation_start=start_date)
        bb = fred.get_series('BAMLH0A1HYBB', observation_start=start_date)
        
        # 2. 新增數據：RRP套利利差 (3個月國債 - RRP利率)
        t3m = fred.get_series('DGS3MO', observation_start=start_date)
        rrp_rate = fred.get_series('RRPONTSYAWARD', observation_start=start_date)

        df = pd.DataFrame({
            'Fed_Assets': fed_assets, 'TGA': tga, 'RRP': rrp,
            'Yield_Curve': yc_10y3m, 'CCC': ccc, 'BB': bb,
            'T3M': t3m, 'RRP_Rate': rrp_rate
        })
        
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
        stock.index = stock.index.tz_localize(None)
        return stock
    except:
        return None

# --- 新增：VPIN 計算引擎 ---
def calculate_vpin(df, bucket_volume, window=50):
    """
    計算 VPIN (訂單流毒性指標)
    """
    df = df.copy()
    # 1. 計算價格變化 (Delta P)
    df['dP'] = df['Close'].diff()
    
    # 2. 估算買賣壓力 (Bulk Volume Classification)
    # 使用常態分佈機率來分配模糊地帶
    sigma = df['dP'].std()
    # 避免 sigma 為 0 的情況
    if sigma == 0: sigma = 0.0001
    
    prob_buy = norm.cdf(df['dP'] / sigma) # 買入機率
    
    df['Buy_Vol'] = df['Volume'] * prob_buy
    df['Sell_Vol'] = df['Volume'] * (1 - prob_buy)
    
    # 3. 將時間序列轉換為「體積序列」 (Volume Bucketing)
    df['Cum_Vol'] = df['Volume'].cumsum()
    # 計算每個 Bar 屬於哪個桶
    df['Bucket_ID'] = (df['Cum_Vol'] // bucket_volume).astype(int)
    
    # 根據桶 ID 聚合數據
    buckets = df.groupby('Bucket_ID').agg({
        'Buy_Vol': 'sum',
        'Sell_Vol': 'sum',
        'Close': 'last', # 記錄桶結束時的價格
        'Datetime': 'last' # 記錄桶結束時的時間
    })
    
    # 4. 計算訂單不平衡 (Order Imbalance)
    buckets['OI'] = (buckets['Buy_Vol'] - buckets['Sell_Vol']).abs()
    
    # 5. 計算 VPIN
    # VPIN = 移動平均(OI) / 移動平均(Total Volume) -> 其實分母就是 bucket_volume (近似)
    # 為了精確，我們用 window 內的總 OI / window 內的總成交量
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

        # 更新 Tabs，加入第四個 VPIN tab
        tab1, tab2, tab3, tab4 = st.tabs(["💧 流動性估值", "📉 殖利率曲線", "🔥 信用利差", "☢️ VPIN 毒性偵測"])

        with tab1:
            st.subheader(f"美元淨流動性 vs {compare_index.split(' ')[0]}")
            
            # 模型訓練
            train_start = f"{reg_start_year}-01-01"
            train_data = merged_df[merged_df.index >= train_start]
            
            if len(train_data) > 30:
                x = train_data['Net_Liquidity']
                y = train_data['Stock_Price']
                slope, intercept = np.polyfit(x, y, 1)
                
                # 計算 R-squared
                correlation_matrix = np.corrcoef(x, y)
                correlation_xy = correlation_matrix[0, 1]
                r_squared = correlation_xy ** 2
                
                merged_df['Fair_Value'] = merged_df['Net_Liquidity'] * slope + intercept
                merged_df['Deviation'] = merged_df['Stock_Price'] - merged_df['Fair_Value']
                merged_df['Deviation_Pct'] = (merged_df['Deviation'] / merged_df['Fair_Value']) * 100
                
                latest = merged_df.iloc[-1]

                # 指標顯示
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("當前淨流動性", f"${latest['Net_Liquidity']:.2f} T")
                c2.metric("理論公允股價", f"{latest['Fair_Value']:.0f}")
                is_bubble = latest['Deviation_Pct'] > 0
                c3.metric("⚠️ 溢價率" if is_bubble else "✅ 折價率", f"{latest['Deviation_Pct']:.1f}%", delta_color="inverse")
                c4.metric("模型可信度 (R²)", f"{r_squared:.2f}", delta_color="normal" if r_squared > 0.7 else "inverse")

                # --- Plotly 交互式圖表 ---
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.03, row_heights=[0.7, 0.3],
                                    subplot_titles=(f"Price vs Liquidity Model ({reg_start_year}-Present)", "Deviation % (Bubble/Discount)"))

                # 上圖
                fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['Stock_Price'], name="Actual Price", line=dict(color='#FFA500', width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['Fair_Value'], name="Fair Value", line=dict(color='#1E90FF', width=2, dash='dash')), row=1, col=1)

                # 下圖
                fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['Deviation_Pct'], name="Deviation %", 
                                         fill='tozeroy', line=dict(color='gray', width=0.5),
                                         fillcolor='rgba(200, 200, 200, 0.2)'), row=2, col=1)
                colors = np.where(merged_df['Deviation_Pct'] > 0, 'rgba(255, 0, 0, 0.5)', 'rgba(0, 255, 0, 0.5)')
                fig.add_trace(go.Bar(x=merged_df.index, y=merged_df['Deviation_Pct'], name="Bubble/Crash", marker_color=colors), row=2, col=1)

                fig.update_layout(height=700, hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", y=1.1), xaxis_rangeslider_visible=False)
                fig.update_yaxes(title_text="Price Index", row=1, col=1)
                fig.update_yaxes(title_text="Deviation (%)", row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)
                st.info("💡 **操作指南：** 使用滑鼠滾輪可縮放時間軸；右上角工具列可選擇「框選放大」或是「重置視圖」。")
            else:
                st.warning("數據不足，無法計算模型。")

        with tab2:
            st.subheader("雙重利差監控：同一參考系比較 (Shared Y-Axis)")
            fig_yc = go.Figure()
            fig_yc.add_trace(go.Scatter(x=df.index, y=df['Yield_Curve'], name="10Y-3M (Macro Cycle)", line=dict(color='#00FFFF', width=2)))
            fig_yc.add_trace(go.Scatter(x=df.index, y=df['Arb_Spread'], name="3M T-Bill - RRP (Liquidity Plumbing)", line=dict(color='#FF00FF', width=2, dash='dot')))
            fig_yc.add_hrect(y0=0, y1=min(df['Yield_Curve'].min(), -1.0), fillcolor="red", opacity=0.15, line_width=0, annotation_text="Recession Zone (Inverted)", annotation_position="bottom right")
            fig_yc.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.8)
            fig_yc.update_layout(height=600, hovermode="x unified", legend=dict(orientation="h", y=1.05), title_text="Spread Comparison (%)", yaxis_title="Spread Strength (Percentage Points)", xaxis_title="Date")
            st.plotly_chart(fig_yc, use_container_width=True)
            st.info("""
            **物理學解讀 (同軸比較):**
            * **振幅差異:** 你會發現 **青線 (宏觀)** 的波動幅度遠大於 **粉紅線 (微觀)**。這是正常的，因為 RRP 套利是極短期的無風險操作，利差通常被壓縮在 0.05% - 0.2% 之間。
            * **危險訊號:** 如果有一天，你看到 **粉紅線 (微觀)** 的波動幅度突然放大，甚至追上了青線的高度，那代表**市場機制失效 (Broken Mechanism)**，那是比經濟衰退更可怕的流動性崩潰。
            """)

        with tab3:
            st.subheader("垃圾債壓力指標 (CCC - BB)")
            fig_cs = go.Figure()
            fig_cs.add_trace(go.Scatter(x=df.index, y=df['Credit_Stress'], name="Credit Stress", fill='tozeroy', line=dict(color='firebrick')))
            fig_cs.update_layout(hovermode="x unified")
            st.plotly_chart(fig_cs, use_container_width=True)

        with tab4:
            st.subheader("☢️ VPIN 訂單流毒性偵測 (微觀結構)")
            st.markdown("""
            **VPIN (Volume-Synchronized Probability of Informed Trading)** 是高頻交易商用來偵測「毒性訂單流」的指標。
            * **原理：** 當 VPIN 飆高，代表市場上出現單邊的大量「知情交易」(Smart Money 正在倒貨或吸籌)，造市商面臨極大風險。
            * **解讀：** * **> 0.6 (橘色)：** 毒性警告，流動性可能開始抽離。
                * **> 0.8 (紅色)：** 極度危險，歷史上多次閃崩 (Flash Crash) 前兆。
            """)

            # 1. 處理代碼映射 (指數通常沒量，需轉為 ETF)
            ticker_map = {
                "^GSPC": "SPY", 
                "RSP": "RSP",
                "^NDX": "QQQ", 
                "^SOX": "SOXX", 
                "BTC-USD": "BTC-USD"
            }
            raw_symbol = compare_index.split(' ')[0]
            vpin_symbol = ticker_map.get(raw_symbol, raw_symbol) # 預設映射，若無則用原代碼

            st.write(f"正在分析標的： **{vpin_symbol}** (使用 1分鐘 K線數據)")
            
            # 2. 觸發按鈕 (避免每次自動跑，因為 1m 數據較慢)
            if st.button("🚀 啟動 VPIN 掃描 (分析過去 5 天)", type="primary"):
                with st.spinner("正在下載高頻數據並計算流體力學..."):
                    try:
                        # 下載數據 (限制 5 天，因為 1m 數據量大且 Yahoo 限制)
                        df_1m = yf.download(vpin_symbol, period='5d', interval='1m', progress=False)
                        
                        if len(df_1m) > 0:
                            # 扁平化 MultiIndex Columns (如果有的話)
                            if isinstance(df_1m.columns, pd.MultiIndex):
                                df_1m.columns = df_1m.columns.get_level_values(0)
                            
                            df_1m = df_1m.reset_index()
                            # 確保有 Datetime 欄位
                            if 'Datetime' not in df_1m.columns:
                                df_1m.rename(columns={'index': 'Datetime'}, inplace=True)
                            
                            # 動態設定 Bucket Size (大約每 15 分鐘的平均量為一個桶)
                            avg_vol_per_min = df_1m['Volume'].mean()
                            dynamic_bucket = int(avg_vol_per_min * 15) 
                            
                            # 計算 VPIN
                            vpin_data = calculate_vpin(df_1m, bucket_volume=dynamic_bucket)
                            
                            # 繪圖
                            fig_vpin = go.Figure()
                            
                            # VPIN 線
                            fig_vpin.add_trace(go.Scatter(
                                x=vpin_data['Datetime'], 
                                y=vpin_data['VPIN'], 
                                name="VPIN Index", 
                                line=dict(color='#00FF00', width=2)
                            ))
                            
                            # 價格線 (副軸，供對照) - 這裡我們簡單化，只畫 VPIN，價格可看其他 Tab
                            # 或者加上一條對照用的價格線 (Normalize 到 0-1) ? 
                            # 為了保持 VPIN 清晰，我們只畫閾值
                            
                            # 閾值線
                            fig_vpin.add_hline(y=0.6, line_dash="dash", line_color="orange", annotation_text="Toxic (0.6)")
                            fig_vpin.add_hline(y=0.8, line_dash="solid", line_color="red", annotation_text="CRASH RISK (0.8)")
                            
                            # 顏色邏輯：VPIN 越高越紅
                            fig_vpin.update_traces(line=dict(color='cyan')) # 預設青色
                            
                            # 新增：動態變色線條 (進階) - 這裡用簡單的區域填色
                            fig_vpin.add_hrect(y0=0.8, y1=1.0, fillcolor="red", opacity=0.2, line_width=0)
                            fig_vpin.add_hrect(y0=0.6, y1=0.8, fillcolor="orange", opacity=0.1, line_width=0)

                            fig_vpin.update_layout(
                                height=500,
                                title=f"VPIN Order Flow Toxicity: {vpin_symbol} (Bucket Size: {dynamic_bucket:,} shares)",
                                yaxis_title="VPIN (0 to 1)",
                                xaxis_title="Time",
                                hovermode="x unified",
                                yaxis_range=[0, 1.0]
                            )
                            
                            st.plotly_chart(fig_vpin, use_container_width=True)
                            
                            latest_vpin = vpin_data['VPIN'].iloc[-1]
                            if latest_vpin > 0.8:
                                st.error(f"🚨 嚴重警告：當前 VPIN = {latest_vpin:.2f}。市場毒性極高，主力正在大量單邊交易，請隨時準備閃崩！")
                            elif latest_vpin > 0.6:
                                st.warning(f"⚠️ 注意：當前 VPIN = {latest_vpin:.2f}。訂單流毒性上升，流動性正在變薄。")
                            else:
                                st.success(f"✅ 安全：當前 VPIN = {latest_vpin:.2f}。市場微觀結構穩定。")
                                
                        else:
                            st.error("無法下載高頻數據，請確認市場是否開盤或代碼是否正確。")
                    except Exception as e:
                        st.error(f"計算發生錯誤: {e}")

else:
    st.info("👈 請在左側輸入 FRED API Key 以啟動交互式戰情室")
