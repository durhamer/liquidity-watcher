import streamlit as st
import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# --- 1. 頁面設定 ---
st.set_page_config(page_title="Alpha 宏觀戰情室 Pro", layout="centered")
st.title("🦅 Alpha 宏觀戰情室 Pro")
st.markdown("監控全球資金水位與市場估值的核心儀表板")

# --- 2. 側邊欄：設定 ---
with st.sidebar:
    st.header("⚙️ 參數設定")
    api_key_input = st.text_input("輸入 FRED API Key", type="password")
    
    st.divider()
    
    # 優化 1: 加入 RSP (等權重) 讓你能一鍵切換
    st.subheader("📈 股市對比")
    compare_index = st.selectbox(
        "選擇指數",
        ["^GSPC (S&P 500 - 七巨頭)", "RSP (S&P 500 等權重 - 真實經濟)", "^NDX (Nasdaq 100)", "^SOX (費半)", "BTC-USD (比特幣)"]
    )
    
    # 優化 2: 增加「回歸分析」的時間區間
    st.subheader("🧮 模型訓練區間")
    st.caption("選擇用哪段時間的數據來定義「正常關係」")
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
        # 1. 淨流動性數據 (Net Liquidity)
        # WALCL: Fed Total Assets
        # WTREGEN: TGA (財政部帳戶)
        # RRPONTSYD: 逆回購 (RRP)
        fed_assets = fred.get_series('WALCL', observation_start=start_date)
        tga = fred.get_series('WTREGEN', observation_start=start_date)
        rrp = fred.get_series('RRPONTSYD', observation_start=start_date)
        
        # 2. 殖利率曲線 & 信用利差
        yc_10y3m = fred.get_series('T10Y3M', observation_start=start_date)
        ccc = fred.get_series('BAMLH0A3HYC', observation_start=start_date)
        bb = fred.get_series('BAMLH0A1HYBB', observation_start=start_date)

        # 合併與清洗
        df = pd.DataFrame({
            'Fed_Assets': fed_assets, 'TGA': tga, 'RRP': rrp,
            'Yield_Curve': yc_10y3m, 'CCC': ccc, 'BB': bb
        })
        df = df.fillna(method='ffill').dropna()
        
        # 計算核心指標
        # 單位換算成「兆 (Trillions)」
        df['Net_Liquidity'] = (df['Fed_Assets'] - df['TGA'] - df['RRP']) / 1000000 
        df['Credit_Stress'] = df['CCC'] - df['BB']
        
        return df
    except Exception as e:
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

# --- 4. 主邏輯 ---
if api_key_input:
    with st.spinner('正在從聯準會與華爾街下載數據...'):
        df = get_macro_data(api_key_input, days_back + 365) # 多抓一點給回歸用
        
    if df is not None:
        stock_series = get_stock_data(compare_index, df.index[0].strftime('%Y-%m-%d'))
        
        # 合併 股市 與 宏觀數據 (取交集)
        merged_df = pd.concat([df, stock_series], axis=1).dropna()
        merged_df.columns = list(df.columns) + ['Stock_Price']

        # --- Tab 分頁 ---
        tab1, tab2, tab3 = st.tabs(["💧 流動性估值模型 (Fair Value)", "📉 殖利率曲線 (衰退)", "🔥 信用利差 (違約)"])

        # ==========================================
        # Tab 1: 流動性估值模型 (物理學家的最愛)
        # ==========================================
        with tab1:
            st.subheader(f"美元淨流動性 vs {compare_index.split(' ')[0]}")
            
            # 1. 訓練回歸模型 (找出物理定律)
            # 篩選出訓練區間的數據
            train_start = f"{reg_start_year}-01-01"
            train_data = merged_df[merged_df.index >= train_start]
            
            if len(train_data) > 30:
                # 準備 X (流動性) 和 Y (股價)
                x = train_data['Net_Liquidity']
                y = train_data['Stock_Price']
                
                # --- 1. 計算線性回歸 (Math) ---
                slope, intercept = np.polyfit(x, y, 1)
                
                # --- 2. 新增：計算 R-squared (測謊儀) ---
                correlation_matrix = np.corrcoef(x, y)
                correlation_xy = correlation_matrix[0, 1]
                r_squared = correlation_xy ** 2
                
                # 計算理論價格
                merged_df['Fair_Value'] = merged_df['Net_Liquidity'] * slope + intercept
                merged_df['Deviation'] = merged_df['Stock_Price'] - merged_df['Fair_Value']
                merged_df['Deviation_Pct'] = (merged_df['Deviation'] / merged_df['Fair_Value']) * 100
                
                latest = merged_df.iloc[-1]

                # --- 3. 顯示診斷數據 (UI Update) ---
                st.markdown("#### 🔬 模型診斷報告")
                d_col1, d_col2, d_col3, d_col4 = st.columns(4)
                
                with d_col1:
                    st.metric("當前淨流動性", f"${latest['Net_Liquidity']:.2f} T")
                
                with d_col2:
                    st.metric("理論公允股價", f"{latest['Fair_Value']:.0f}")
                
                with d_col3:
                    # 顏色邏輯：泡沫(紅) / 折價(綠)
                    is_bubble = latest['Deviation_Pct'] > 0
                    st.metric(
                        "⚠️ 溢價率 (泡沫)" if is_bubble else "✅ 折價率 (低估)", 
                        f"{latest['Deviation_Pct']:.1f}%", 
                        f"{latest['Deviation']:.0f} pts",
                        delta_color="inverse"
                    )
                
                with d_col4:
                    # 顏色邏輯：R²高(綠=可信) / R²低(紅=不可信)
                    r2_color = "normal"
                    if r_squared > 0.7: r2_color = "off" # 綠色/灰色 (Streamlit normal is good)
                    elif r_squared < 0.3: r2_color = "inverse" # 紅色 (Warning)
                    
                    st.metric(
                        "📊 模型可信度 (R²)", 
                        f"{r_squared:.2f}",
                        "越接近 1 越準確",
                        delta_color=r2_color
                    )

                # 如果 R² 太低，顯示警告
                if r_squared < 0.3:
                    st.warning(f"🚨 **注意：** 此資產與流動性的相關性極低 (R²={r_squared:.2f})。這代表它的漲跌主要**不是**由資金面驅動的（可能是基本面或避險情緒）。模型算出的「溢價」參考價值不高。")


                # 繪圖 1: 走勢對比
                fig, ax1 = plt.subplots(figsize=(10, 6))
                
                # 畫公允價值區間 (Fair Value Band)
                ax1.plot(merged_df.index, merged_df['Stock_Price'], color='orange', label='Actual Price', linewidth=2)
                ax1.plot(merged_df.index, merged_df['Fair_Value'], color='blue', linestyle='--', label='Fair Value (Liquidity Model)', alpha=0.7)
                
                # 填色：溢價(紅) vs 折價(綠)
                ax1.fill_between(merged_df.index, merged_df['Stock_Price'], merged_df['Fair_Value'], 
                                 where=(merged_df['Stock_Price'] > merged_df['Fair_Value']), 
                                 color='red', alpha=0.3, label='Overvalued (Bubble)')
                
                ax1.fill_between(merged_df.index, merged_df['Stock_Price'], merged_df['Fair_Value'], 
                                 where=(merged_df['Stock_Price'] <= merged_df['Fair_Value']), 
                                 color='green', alpha=0.3, label='Undervalued')

                ax1.set_ylabel("Price")
                ax1.set_title("Market Price vs Liquidity-Implied Fair Value")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # 繪圖 2: 散佈圖 (Scatter Plot) - 驗證相關性
                with st.expander("查看相關性物理模型 (Scatter Plot)"):
                    fig2, ax2 = plt.subplots()
                    ax2.scatter(merged_df['Net_Liquidity'], merged_df['Stock_Price'], alpha=0.5, c=merged_df.index.year, cmap='viridis')
                    # 畫出回歸線
                    x_seq = np.linspace(merged_df['Net_Liquidity'].min(), merged_df['Net_Liquidity'].max(), 100)
                    y_seq = slope * x_seq + intercept
                    ax2.plot(x_seq, y_seq, 'r--', label='Regression Line')
                    
                    ax2.set_xlabel("Net Liquidity (Trillions)")
                    ax2.set_ylabel("Stock Index Price")
                    ax2.legend()
                    st.pyplot(fig2)
                    st.caption("顏色代表年份。如果點都在紅線上方，代表脫離基本面。")

            else:
                st.warning("數據不足，無法計算模型。請調整回歸起始年。")

        # ==========================================
        # Tab 2: 殖利率曲線
        # ==========================================
        with tab2:
            st.subheader("10年期 - 3個月公債利差")
            latest_yc = df['Yield_Curve'].iloc[-1]
            st.metric("10Y-3M 利差", f"{latest_yc:.2f}%")
            
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            ax3.axhline(y=0, color='black', linewidth=1)
            ax3.plot(df.index, df['Yield_Curve'], color='black', linewidth=1)
            ax3.fill_between(df.index, df['Yield_Curve'], 0, where=(df['Yield_Curve'] < 0), color='red', alpha=0.3)
            ax3.fill_between(df.index, df['Yield_Curve'], 0, where=(df['Yield_Curve'] > 0), color='green', alpha=0.3)
            st.pyplot(fig3)

        # ==========================================
        # Tab 3: 信用利差
        # ==========================================
        with tab3:
            st.subheader("垃圾債壓力指標 (CCC - BB)")
            st.line_chart(df['Credit_Stress'])

else:
    st.info("👈 請在左側輸入 FRED API Key 以啟動戰情室")
