import streamlit as st
import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- 網頁設定 ---
st.set_page_config(page_title="流動性壓力監控", layout="centered")
st.title("🌊 市場流動性壓力監控 (CCC vs BB)")

# --- 側邊欄：設定 ---
with st.sidebar:
    st.header("設定")
    # 這裡讓你在網頁上輸入 API Key，或是從 Secrets 讀取
    api_key_input = st.text_input("輸入 FRED API Key", type="password")
    days_back = st.slider("回溯天數", min_value=180, max_value=3650, value=730, step=30)
    st.markdown("[申請 FRED API Key](https://fred.stlouisfed.org/docs/api/api_key.html)")

# --- 主程式邏輯 ---
def get_data(api_key, days):
    fred = Fred(api_key=api_key)
    start_date = datetime.now() - timedelta(days=days)
    
    with st.spinner('正在從聯準會資料庫抓取數據...'):
        try:
            ccc_spread = fred.get_series('BAMLH0A3HYC', observation_start=start_date)
            bb_spread = fred.get_series('BAMLH0A1HYBB', observation_start=start_date)
            
            df = pd.DataFrame({'CCC_OAS': ccc_spread, 'BB_OAS': bb_spread})
            df.dropna(inplace=True)
            df['Stress_Signal'] = df['CCC_OAS'] - df['BB_OAS']
            return df
        except Exception as e:
            st.error(f"抓取數據失敗: {e}")
            return None

# --- 執行與顯示 ---
if api_key_input:
    df = get_data(api_key_input, days_back)

    if df is not None and not df.empty:
        latest = df.iloc[-1]
        prev_month = df.iloc[-30] if len(df) > 30 else df.iloc[0]
        change = latest['Stress_Signal'] - prev_month['Stress_Signal']

        # 1. 顯示關鍵指標 (大數字儀表板)
        st.subheader(f"📅 數據日期: {df.index[-1].strftime('%Y-%m-%d')}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CCC 級利差 (爛)", f"{latest['CCC_OAS']:.2f}%")
        with col2:
            st.metric("BB 級利差 (穩)", f"{latest['BB_OAS']:.2f}%")
        with col3:
            st.metric("⚠️ 壓力指標 (差值)", 
                      f"{latest['Stress_Signal']:.2f}%", 
                      f"{change:+.2f}% (月變化)",
                      delta_color="inverse") # 數值變大會顯示紅色(危險)

        # 2. 警報邏輯
        stress_val = latest['Stress_Signal']
        if stress_val > 6.0:
            st.error("🚨 **紅色警報**：垃圾債市場裂痕嚴重！聰明錢正在撤離！")
        elif change > 0.5:
            st.warning("⚠️ **注意**：壓力指標正在快速擴大 (趨勢轉壞)")
        else:
            st.success("✅ **狀態**：目前市場情緒尚屬穩定 (或過度自滿)")

        # 3. 繪圖
        st.subheader("趨勢圖表")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # 畫 CCC 和 BB
        ax.plot(df.index, df['CCC_OAS'], label='CCC (High Risk)', color='red', alpha=0.3, linestyle='--')
        ax.plot(df.index, df['BB_OAS'], label='BB (Safe-ish)', color='green', alpha=0.3, linestyle='--')
        
        # 畫壓力指標
        ax.plot(df.index, df['Stress_Signal'], label='Stress Signal', color='blue', linewidth=2)
        ax.fill_between(df.index, df['Stress_Signal'], 0, color='blue', alpha=0.1)
        
        ax.set_title('Liquidity Stress Monitor')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 將 Matplotlib 圖表顯示在網頁上
        st.pyplot(fig)

        # 4. 顯示原始數據表格 (可選)
        with st.expander("查看原始數據"):
            st.dataframe(df.sort_index(ascending=False))

else:
    st.info("👈 請在左側輸入你的 FRED API Key 以開始分析")
