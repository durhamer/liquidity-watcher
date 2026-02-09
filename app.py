import streamlit as st
import pandas as pd
from fredapi import Fred
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go # 引入 Plotly 交互式圖表庫
from plotly.subplots import make_subplots

# --- 1. 頁面設定 ---
st.set_page_config(page_title="Alpha 宏觀戰情室 Pro (Interactive)", layout="wide") # 改成寬版配置
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

# --- 3. 數據核心 (不變) ---
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
        
        # 新增計算：套利利差 (正值代表資金會從 RRP 流出買國債)
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

# --- 4. 繪圖函數 (Plotly 核心) ---
def plot_interactive_chart(df, ticker_name):
    # 建立雙軸圖表
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. 畫股價 (實際值)
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Stock_Price'], name=f"{ticker_name} Price", line=dict(color='orange', width=2)),
        secondary_y=False,
    )

    # 2. 畫公允價值 (理論值)
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Fair_Value'], name="Fair Value (Liquidity)", line=dict(color='blue', width=2, dash='dash')),
        secondary_y=False,
    )

    # 3. 畫綠色區域 (折價/低估) - 使用 fill='tonexty' 技巧
    # 這裡我們需要一點技巧來畫填色區域，Plotly 沒有 matplotlib 的 fill_between 那麼直觀
    # 但為了交互性，我們用簡單的方式：只畫線，或者用更進階的 shape。
    # 為了保持效能，這裡我們用散佈點的顏色來輔助，或者直接畫差異柱狀圖在下方。
    
    # 改進方案：我們把「泡沫/折價」畫成下方的柱狀圖，這樣更清楚
    
    return fig

# --- 5. 主邏輯 ---
if api_key_input:
    with st.spinner('正在初始化量子數據鏈接...'):
        df = get_macro_data(api_key_input, days_back + 365)
        
    if df is not None:
        stock_series = get_stock_data(compare_index, df.index[0].strftime('%Y-%m-%d'))
        merged_df = pd.concat([df, stock_series], axis=1).dropna()
        merged_df.columns = list(df.columns) + ['Stock_Price']

        tab1, tab2, tab3 = st.tabs(["💧 流動性估值 (Interactive)", "📉 殖利率曲線", "🔥 信用利差"])

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
                
                # 建立主圖 (上) 和 副圖 (下 - 溢價率)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.03, row_heights=[0.7, 0.3],
                                    subplot_titles=(f"Price vs Liquidity Model ({reg_start_year}-Present)", "Deviation % (Bubble/Discount)"))

                # 上圖：股價 vs 公允價值
                fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['Stock_Price'], name="Actual Price", line=dict(color='#FFA500', width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['Fair_Value'], name="Fair Value", line=dict(color='#1E90FF', width=2, dash='dash')), row=1, col=1)

                # 下圖：溢價率 (Area Chart)
                # 分開畫正值(紅)和負值(綠)
                fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['Deviation_Pct'], name="Deviation %", 
                                         fill='tozeroy', line=dict(color='gray', width=0.5),
                                         fillcolor='rgba(200, 200, 200, 0.2)'), row=2, col=1)

                # 用顏色區分紅綠
                colors = np.where(merged_df['Deviation_Pct'] > 0, 'rgba(255, 0, 0, 0.5)', 'rgba(0, 255, 0, 0.5)')
                fig.add_trace(go.Bar(x=merged_df.index, y=merged_df['Deviation_Pct'], name="Bubble/Crash", marker_color=colors), row=2, col=1)

                # 更新佈局
                fig.update_layout(
                    height=700, #圖表高度
                    hovermode="x unified", # 鼠標懸停顯示所有數據
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", y=1.1),
                    xaxis_rangeslider_visible=False # 隱藏底部的滑條，因為我們可以直接滾輪縮放
                )
                
                # 設定 Y 軸標題
                fig.update_yaxes(title_text="Price Index", row=1, col=1)
                fig.update_yaxes(title_text="Deviation (%)", row=2, col=1)

                # 顯示圖表
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 **操作指南：** 使用滑鼠滾輪可縮放時間軸；右上角工具列可選擇「框選放大」或是「重置視圖」。")

            else:
                st.warning("數據不足，無法計算模型。")

        with tab2:
            st.subheader("雙重利差監控：經濟衰退 vs. 資金套利")
            
            # 建立雙軸圖表 (雖然單位都是%，但雙軸可以避免互相干擾視覺)
            fig_yc = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 1. 主線：10年期 - 3個月 (經濟衰退指標) - 青色
            fig_yc.add_trace(go.Scatter(
                x=df.index, 
                y=df['Yield_Curve'], 
                name="10Y-3M (Recession Indicator)", 
                line=dict(color='#00FFFF', width=2)
            ), secondary_y=False)
            
            # 2. 副線：3個月 - RRP利率 (RRP提款指標) - 粉紅色虛線
            fig_yc.add_trace(go.Scatter(
                x=df.index, 
                y=df['Arb_Spread'], 
                name="3M T-Bill - RRP (Liquidity Drain)", 
                line=dict(color='#FF00FF', width=2, dash='dot')
            ), secondary_y=True) # 放在右軸，或者為了比較也可以放左軸(secondary_y=False)，看你喜好
            
            # 3. 裝飾：衰退訊號區 (10Y-3M < 0)
            fig_yc.add_hrect(
                y0=0, y1=min(df['Yield_Curve'].min(), -1), 
                fillcolor="red", opacity=0.1, line_width=0, 
                annotation_text="Recession Zone", secondary_y=False
            )
            
            # 4. 裝飾：套利逆轉區 (3M < RRP)
            # 當這條粉紅線跌破 0，代表 RRP 開始吸血 (危機信號)
            fig_yc.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)

            fig_yc.update_layout(
                height=600,
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1),
                title_text="Cyan: Economic Cycle | Magenta: Plumbing Pressure"
            )
            
            # 設定座標軸標題
            fig_yc.update_yaxes(title_text="10Y-3M Spread (%)", secondary_y=False)
            fig_yc.update_yaxes(title_text="3M-RRP Spread (%)", secondary_y=True, showgrid=False)
            
            st.plotly_chart(fig_yc, use_container_width=True)
            
            st.info("""
            **解讀指南 (Physics of Spreads):**
            * 🔵 **青線 (10Y-3M):** 跌入紅色區域 = **經濟衰退倒數**。
            * 🟣 **粉紅線 (3M-RRP):** * **正值 (+):** 資金從 RRP 流出買國債 (流動性釋放/中性)。
                * **負值 (-):** 資金逃回 RRP 避險 (流動性猝死/銀行危機)。**如果這條線急墜破 0，快跑！**
            """)

        with tab3:
            st.subheader("垃圾債壓力指標 (CCC - BB)")
            fig_cs = go.Figure()
            fig_cs.add_trace(go.Scatter(x=df.index, y=df['Credit_Stress'], name="Credit Stress", fill='tozeroy', line=dict(color='firebrick')))
            fig_cs.update_layout(hovermode="x unified")
            st.plotly_chart(fig_cs, use_container_width=True)

else:
    st.info("👈 請在左側輸入 FRED API Key 以啟動交互式戰情室")
