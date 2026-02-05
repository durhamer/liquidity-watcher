import streamlit as st
import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# --- 網頁設定 ---
st.set_page_config(page_title="流動性壓力監控 Pro", layout="centered")
st.title("🌊 市場流動性壓力監控 Pro")

# --- 側邊欄：設定 ---
with st.sidebar:
    st.header("⚙️ 設定面板")
    api_key_input = st.text_input("輸入 FRED API Key", type="password")
    
    st.divider()
    
    # 新增：股市對比功能
    st.subheader("📈 股市疊圖對比")
    compare_index = st.selectbox(
        "選擇要對比的指數",
        ["None (不對比)", "^GSPC (標普500)", "^NDX (納斯達克100)", "^SOX (費城半導體)"]
    )
    
    days_back = st.slider("回溯天數", min_value=180, max_value=3650, value=730, step=30)
    st.markdown("---")
    st.markdown("[申請 FRED API Key](https://fred.stlouisfed.org/docs/api/api_key.html)")

# --- 核心邏輯 ---
def get_fred_data(api_key, days):
    fred = Fred(api_key=api_key)
    start_date = datetime.now() - timedelta(days=days)
    
    with st.spinner('正在從聯準會 (FRED) 抓取信貸數據...'):
        try:
            ccc_spread = fred.get_series('BAMLH0A3HYC', observation_start=start_date)
            bb_spread = fred.get_series('BAMLH0A1HYBB', observation_start=start_date)
            
            df = pd.DataFrame({'CCC_OAS': ccc_spread, 'BB_OAS': bb_spread})
            df.dropna(inplace=True)
            df['Stress_Signal'] = df['CCC_OAS'] - df['BB_OAS']
            return df
        except Exception as e:
            st.error(f"FRED 數據抓取失敗: {e}")
            return None

def get_stock_data(ticker, start_date):
    if ticker.startswith("None"):
        return None
    
    symbol = ticker.split(" ")[0] # 取出代號部分
    with st.spinner(f'正在從 Yahoo Finance 抓取 {symbol} 數據...'):
        try:
            stock = yf.download(symbol, start=start_date, progress=False)
            # 確保時區單純化，避免與 FRED 數據合併時報錯
            stock.index = stock.index.tz_localize(None) 
            return stock['Close']
        except Exception as e:
            st.warning(f"股市數據抓取失敗: {e}")
            return None

# --- 執行與顯示 ---
if api_key_input:
    # 1. 抓取 FRED 數據
    df_fred = get_fred_data(api_key_input, days_back)

    if df_fred is not None and not df_fred.empty:
        # 計算日期範圍供股市數據使用
        start_date_str = df_fred.index[0].strftime('%Y-%m-%d')
        
        # 2. 抓取股市數據 (如果有選)
        stock_series = get_stock_data(compare_index, start_date_str)

        # 3. 顯示儀表板
        latest = df_fred.iloc[-1]
        prev_month = df_fred.iloc[-30] if len(df_fred) > 30 else df_fred.iloc[0]
        change = latest['Stress_Signal'] - prev_month['Stress_Signal']

        st.subheader(f"📅 數據日期: {df_fred.index[-1].strftime('%Y-%m-%d')}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CCC 級利差 (爛)", f"{latest['CCC_OAS']:.2f}%")
        with col2:
            st.metric("BB 級利差 (穩)", f"{latest['BB_OAS']:.2f}%")
        with col3:
            st.metric("⚠️ 壓力指標", 
                      f"{latest['Stress_Signal']:.2f}%", 
                      f"{change:+.2f}% (月變動)",
                      delta_color="inverse")

        # 警報區
        if latest['Stress_Signal'] > 6.0:
            st.error("🚨 **紅色警報**：垃圾債市場裂痕嚴重！聰明錢正在撤離！")
        elif change > 0.5:
            st.warning("⚠️ **注意**：壓力指標正在快速擴大 (趨勢轉壞)")

        # 4. 繪製雙軸圖表
        st.subheader("趨勢對比圖")
        
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # 左軸：繪製壓力指標 (藍色)
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stress Signal (Spread %)', color=color)
        ax1.plot(df_fred.index, df_fred['Stress_Signal'], color=color, linewidth=2, label='Stress Signal')
        ax1.fill_between(df_fred.index, df_fred['Stress_Signal'], 0, color=color, alpha=0.1)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)

        # 右軸：繪製股市指數 (橘色) - 如果有選的話
        if stock_series is not None and not stock_series.empty:
            ax2 = ax1.twinx()  # 建立共享 X 軸的第二 Y 軸
            color_stock = 'tab:orange'
            stock_name = compare_index.split(" ")[1] # 取得中文名稱
            ax2.set_ylabel(f'{stock_name} Price', color=color_stock)
            ax2.plot(stock_series.index, stock_series, color=color_stock, linewidth=2, linestyle='--', label=stock_name)
            ax2.tick_params(axis='y', labelcolor=color_stock)
            
            # 在圖上標示圖例
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        else:
            ax1.legend(loc='upper left')

        st.pyplot(fig)

        with st.expander("查看 FRED 原始數據"):
            st.dataframe(df_fred.sort_index(ascending=False))

else:
    st.info("👈 請在左側輸入 FRED API Key 開始")
