import streamlit as st
import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# --- 網頁設定 ---
st.set_page_config(page_title="Alpha 宏觀戰情室", layout="centered")
st.title("🦅 Alpha 宏觀戰情室")

# --- 側邊欄：設定 ---
with st.sidebar:
    st.header("⚙️ 設定面板")
    api_key_input = st.text_input("輸入 FRED API Key", type="password")
    
    st.divider()
    
    st.subheader("📈 股市疊圖對比")
    compare_index = st.selectbox(
    "選擇要對比的指數",
    [
        "None (不對比)", 
        "^GSPC (S&P 500 - 被七巨頭扭曲)", 
        "RSP (S&P 500 等權重 - 真實經濟)",   # <--- 加入這個
        "^NDX (Nasdaq 100)", 
        "^SOX (Phlx Semi)"
    ]
)
    
    days_back = st.slider("回溯天數", min_value=365, max_value=3650, value=1095, step=30)
    st.info("建議回溯天數設為 1095 (3年) 以上，較能看清週期。")

# --- 數據抓取函數 ---
@st.cache_data(ttl=3600) # 快取 1 小時，避免重複抓取
def get_fred_data(api_key, days):
    fred = Fred(api_key=api_key)
    start_date = datetime.now() - timedelta(days=days)
    
    try:
        # 1. 信用利差數據
        ccc = fred.get_series('BAMLH0A3HYC', observation_start=start_date)
        bb = fred.get_series('BAMLH0A1HYBB', observation_start=start_date)
        
        # 2. 淨流動性數據 (Net Liquidity)
        # WALCL: Fed Total Assets (週資料)
        # WTREGEN: Treasury General Account (TGA) (週資料)
        # RRPONTSYD: Overnight Reverse Repo (RRP) (日資料)
        fed_assets = fred.get_series('WALCL', observation_start=start_date)
        tga = fred.get_series('WTREGEN', observation_start=start_date)
        rrp = fred.get_series('RRPONTSYD', observation_start=start_date)
        
        # 3. 殖利率曲線 (Yield Curve)
        # T10Y3M: 10-Year Minus 3-Month Treasury Yield Spread
        yc_10y3m = fred.get_series('T10Y3M', observation_start=start_date)

        # 整理數據
        df = pd.DataFrame({
            'CCC': ccc, 'BB': bb, 
            'Fed_Assets': fed_assets, 'TGA': tga, 'RRP': rrp,
            'Yield_Curve': yc_10y3m
        })
        
        # 處理頻率不一致問題 (RRP是日更，其他是週更，用 ffill 填補)
        df = df.fillna(method='ffill').dropna()
        
        # 計算衍生指標
        df['Stress_Signal'] = df['CCC'] - df['BB']
        # 淨流動性 = Fed資產 - TGA - RRP (單位轉換為兆美元)
        df['Net_Liquidity'] = (df['Fed_Assets'] - df['TGA'] - df['RRP']) / 1000000 
        
        return df
    except Exception as e:
        return None

def get_stock_data(ticker, start_date):
    if ticker.startswith("None"):
        return None
    symbol = ticker.split(" ")[0]
    try:
        stock = yf.download(symbol, start=start_date, progress=False)
        stock.index = stock.index.tz_localize(None)
        return stock['Close']
    except:
        return None

# --- 主程式 ---
if api_key_input:
    with st.spinner('正在從聯準會與華爾街抓取最新數據...'):
        df = get_fred_data(api_key_input, days_back)
        
    if df is not None:
        stock_data = get_stock_data(compare_index, df.index[0].strftime('%Y-%m-%d'))
        
        # 使用 Tabs 分頁
        tab1, tab2, tab3 = st.tabs(["💧 美元淨流動性 (最敏感)", "📉 殖利率曲線 (衰退指標)", "🔥 信用利差 (舊版)"])

        # --- Tab 1: 淨流動性 (Net Liquidity) ---
        with tab1:
            st.subheader("美元淨流動性 vs 股市")
            st.markdown("""
            **公式：** `Fed資產負債表 - TGA帳戶 - 逆回購(RRP)`
            \n**解讀：** 這是股市的「燃料」。如果藍線(錢)往下掉，橘線(股市)通常會在 2-4 週後跟著掉。
            """)
            
            latest_liq = df['Net_Liquidity'].iloc[-1]
            prev_liq = df['Net_Liquidity'].iloc[-30]
            delta_liq = latest_liq - prev_liq
            
            st.metric("當前市場淨流動性 (兆美元)", f"${latest_liq:.2f} T", f"{delta_liq:+.2f} T")

            fig, ax1 = plt.subplots(figsize=(10, 5))
            color = 'tab:blue'
            ax1.set_ylabel('Net Liquidity (Trillions $)', color=color)
            ax1.plot(df.index, df['Net_Liquidity'], color=color, linewidth=2, label='Net Liquidity')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)
            
            if stock_data is not None:
                ax2 = ax1.twinx()
                color_stock = 'tab:orange'
                ax2.set_ylabel(f'{compare_index.split(" ")[1]} Price', color=color_stock)
                ax2.plot(stock_data.index, stock_data, color=color_stock, linestyle='--', label='Stock Index')
                ax2.tick_params(axis='y', labelcolor=color_stock)
            
            st.pyplot(fig)

        # --- Tab 2: 殖利率曲線 (Yield Curve) ---
        with tab2:
            st.subheader("10年期 - 3個月公債利差")
            st.markdown("""
            **解讀：** * **倒掛 (0以下)**：預警未來一年內可能衰退。
            * **解除倒掛 (回到0以上)**：**最危險的時刻！** 通常崩盤都發生在「曲線重新變陡、回到正數」的那一瞬間。
            """)
            
            latest_yc = df['Yield_Curve'].iloc[-1]
            st.metric("10Y-3M 利差", f"{latest_yc:.2f}%", delta_color="normal")
            
            if latest_yc > -0.2 and latest_yc < 0.2:
                st.warning("⚠️ 警告：殖利率曲線即將「解除倒掛」，這是崩盤前的經典訊號！")

            fig2, ax = plt.subplots(figsize=(10, 5))
            # 繪製 0 軸線 (危險分界線)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            
            # 根據正負值填色
            ax.plot(df.index, df['Yield_Curve'], color='black', linewidth=1)
            ax.fill_between(df.index, df['Yield_Curve'], 0, where=(df['Yield_Curve'] < 0), color='red', alpha=0.3, label='Inverted (Recession Warning)')
            ax.fill_between(df.index, df['Yield_Curve'], 0, where=(df['Yield_Curve'] > 0), color='green', alpha=0.3, label='Normal')
            
            ax.set_ylabel('Spread (%)')
            ax.grid(True, alpha=0.3)
            
            if stock_data is not None:
                ax3 = ax.twinx()
                ax3.plot(stock_data.index, stock_data, color='tab:orange', linestyle='--', alpha=0.6)
            
            st.pyplot(fig2)

        # --- Tab 3: 信用利差 (Original) ---
        with tab3:
            st.subheader("垃圾債壓力指標 (CCC - BB)")
            st.line_chart(df['Stress_Signal'])
            st.write("這是你原本使用的指標，適合用來確認「現在是不是已經失控」。")

else:
    st.info("👈 請在左側輸入 FRED API Key 以解鎖戰情室")
