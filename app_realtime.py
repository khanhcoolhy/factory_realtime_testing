import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import requests
import random
from datetime import datetime, timedelta
from supabase import create_client

# ===============================================================
# 1. CẤU HÌNH & KẾT NỐI (LOGIC CŨ - GIỮ NGUYÊN)
# ===============================================================
st.set_page_config(page_title="Stanley Factory Monitor", layout="wide", page_icon="🏭")

# --- CSS: Làm đẹp & Ẩn nút mặc định của Streamlit ---
st.markdown("""
<style>
    .status-ok { background-color: #d1e7dd; color: #0f5132; padding: 4px 12px; border-radius: 20px; font-weight: 600; border: 1px solid #badbcc; display: inline-block; }
    .status-err { background-color: #f8d7da; color: #842029; padding: 4px 12px; border-radius: 20px; font-weight: 600; border: 1px solid #f5c2c7; display: inline-block; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #333; }
    h3 { font-size: 1.2rem !important; font-weight: 700 !important; color: #444; }
    .blink_me { animation: blinker 1.5s linear infinite; color: red; font-weight: bold;}
    @keyframes blinker { 50% { opacity: 0; } }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "lstm_factory_v2.pth"
SCALER_PATH = "robust_scaler_v2.pkl"
CONFIG_PATH = "model_config_v2.pkl"
DEVICES = ["4417930D77DA", "AC0BFBCE8797"]
REFRESH_RATE = 2 

try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
except:
    st.error("❌ Thiếu Secrets!")
    st.stop()

@st.cache_resource
def init_connection():
    return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_connection()

@st.cache_resource
def load_ai():
    if not os.path.exists(MODEL_PATH): return None, None, None
    try:
        cfg = joblib.load(CONFIG_PATH); scl = joblib.load(SCALER_PATH)
        class LSTM(nn.Module):
            def __init__(self, n, h=128): super().__init__(); self.l = nn.LSTM(n, h, 3, batch_first=True); self.f = nn.Linear(h, n)
            def forward(self, x): o, _ = self.l(x); return self.f(o[:, -1, :])
        mdl = LSTM(cfg['n_features'], cfg['hidden_dim'])
        mdl.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')); mdl.eval()
        return mdl, scl, cfg
    except: return None, None, None

model, scaler, config = load_ai()

if 'status' not in st.session_state:
    st.session_state.status = {d: False for d in DEVICES}
    st.session_state.buffer = {d: 0 for d in DEVICES}
    st.session_state.logs = {d: [] for d in DEVICES}

# --- HELPERS ---
def send_telegram(msg):
    if not TELEGRAM_TOKEN: return
    try: requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=2)
    except: pass

def get_action(speed):
    if speed < 50: return "Kiểm tra nguồn điện"
    if speed > 10000: return "Kiểm tra biến tần"
    return "Bôi trơn trục"

# --- HÀM BƠM DỮ LIỆU GIẢ (DEMO MODE) ---
def inject_demo_data():
    now_iso = datetime.now().isoformat()
    # Giả lập số liệu ngẫu nhiên
    for i, dev_id in enumerate(DEVICES):
        speed = random.randint(100, 150)
        temp = random.uniform(30, 45)
        payload = {
            "time": now_iso, "DevAddr": dev_id, "Channel": f"0{i+1}",
            "Actual": random.randint(1000000, 2000000), "Status": 1,
            "RunTime": 50000, "HeldTime": 20000,
            "Speed": float(speed), "d_RunTime": 20.0, "d_HeldTime": 0.0,
            "Temp": temp, "Humidity": 70.0
        }
        try: supabase.table("sensor_data").insert(payload).execute()
        except: pass
    st.toast("🚀 Đã bơm dữ liệu mẫu! Biểu đồ sẽ nhảy ngay lập tức.")

# --- FIX LỖI THỜI GIAN TRONG HÀM LẤY DATA (QUAN TRỌNG) ---
def get_recent_data(limit=100):
    try:
        response = supabase.table("sensor_data").select("*").order("time", desc=True).limit(limit).execute()
        df = pd.DataFrame(response.data)
        
        if not df.empty:
            # FIX LỖI: Dùng pd.to_datetime(..., utc=True) và errors='coerce' để xử lý đa định dạng
            df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce') 
            df = df.dropna(subset=['time']) # Loại bỏ dòng lỗi
            
            # Chuyển múi giờ về Asia/Bangkok (+07)
            df['time'] = df['time'].dt.tz_convert('Asia/Bangkok').dt.tz_localize(None)
            
            # Giới hạn 12h gần nhất (giúp biểu đồ trôi mượt và không bị méo)
            cutoff = datetime.now() - timedelta(hours=12)
            df = df[df['time'] > cutoff]
            
        return df
    except Exception as e: 
        # Nếu vẫn lỗi, in ra log để debug
        print(f"Lỗi tải dữ liệu: {e}")
        return pd.DataFrame()

# ===============================================================
# 2. UI COMPONENTS (CHARTS)
# ===============================================================

def create_gauge(value, title, max_val=300, color="green"):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = value,
        title = {'text': title, 'font': {'size': 16, 'color': '#555'}},
        gauge = {
            'axis': {'range': [None, max_val]}, 'bar': {'color': color},
            'bgcolor': "white", 'borderwidth': 1, 'bordercolor': "#ddd",
            'steps': [{'range': [0, max_val*0.7], 'color': '#f0fff4'}, {'range': [max_val*0.7, max_val], 'color': '#ffebee'}],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': max_val * 0.9}
        }
    ))
    fig.update_layout(height=180, margin=dict(t=30,b=10,l=25,r=25))
    return fig

# --- BIỂU ĐỒ CHẠY LIÊN TỤC (SLIDING WINDOW) ---
def create_trend_chart(df, dev_name):
    now = datetime.now()
    x_range = [now - timedelta(minutes=15), now] 

    fig = go.Figure()
    
    if not df.empty:
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['Speed'], fill='tozeroy', mode='lines+markers',
            line=dict(width=2, color='#0ea5e9'), fillcolor='rgba(14, 165, 233, 0.1)', name='Tốc độ'
        ))
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['Temp'], mode='lines', line=dict(color='#f97316', dash='dot', width=1),
            yaxis='y2', name='Nhiệt độ'
        ))

    fig.update_layout(
        title=dict(text="Diễn biến 15 phút qua", font=dict(size=14, color="#555")),
        height=250, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(
            showgrid=False, tickformat='%H:%M:%S', 
            range=x_range, # --- Đảm bảo biểu đồ luôn trôi ---
            fixedrange=True 
        ),
        yaxis=dict(title="Speed", showgrid=True, gridcolor='#f0f0f0', range=[0, 350]),
        yaxis2=dict(title="Temp", overlaying='y', side='right', showgrid=False, range=[0, 100]),
        legend=dict(orientation="h", y=1.1, x=1), plot_bgcolor='white', hovermode="x unified"
    )
    return fig

# ===============================================================
# 3. REAL-TIME TAB
# ===============================================================
def render_realtime_tab():
    with st.sidebar:
        st.header("🎮 Điều khiển Demo")
        if st.button("⚡ Bơm dữ liệu ngay (Demo)", type="primary", use_container_width=True):
            inject_demo_data()
        st.info("Sử dụng nút này để xem hiệu ứng nhảy số ngay lập tức.")

    c1, c2 = st.columns([3, 1])
    c1.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')} | Auto-scroll: ON")
    c2.markdown('<span class="blink_me">● LIVE CONNECTED</span>', unsafe_allow_html=True)
    
    @st.fragment(run_every=REFRESH_RATE)
    def update_loop():
        df_all = get_recent_data(200)
        
        col1, col2 = st.columns(2)
        cols_map = {DEVICES[0]: col1, DEVICES[1]: col2}

        if df_all.empty:
            for dev in DEVICES:
                with cols_map[dev]: st.warning("⏳ Đang chờ Worker bơm dữ liệu mới...")
            return

        for dev in DEVICES:
            df = df_all[df_all['DevAddr'] == dev].sort_values('time')
            if df.empty:
                with cols_map[dev]: st.info("Chưa có dữ liệu cho máy này trong 12h qua.")
                continue
            
            last = df.iloc[-1]
            current_col = cols_map[dev]
            
            # AI Logic (Rút gọn)
            score = 0.0; is_danger = False
            
            if st.session_state.status[dev]: gauge_color = "#ef4444"
            else: gauge_color = "#10b981"

            # RENDER UI
            with current_col:
                with st.container(border=True):
                    h1, h2 = st.columns([3, 1])
                    h1.subheader(f"📡 Device: {dev[-4:]}")
                    h2.markdown(f'<div class="status-ok">RUNNING</div>', unsafe_allow_html=True)

                    st.markdown("---")
                    g1, g2 = st.columns(2)
                    g1.plotly_chart(create_gauge(last['Speed'], "Tốc độ (sp/p)", 300, gauge_color), use_container_width=True, key=f"g_s_{dev}")
                    g2.plotly_chart(create_gauge(last['Temp'], "Nhiệt độ (°C)", 100, "#f59e0b"), use_container_width=True, key=f"g_t_{dev}")

                    m1, m2 = st.columns(2)
                    m1.metric("Sản lượng", f"{last['Actual']:,}")
                    m2.metric("Thời gian chạy", f"{int(last['RunTime']/60)}m")

                    st.markdown("---")
                    fig_trend = create_trend_chart(df, dev)
                    st.plotly_chart(fig_trend, use_container_width=True, key=f"trend_{dev}")

    update_loop()

# ===============================================================
# 4. REPORT TAB (FIX LỖI THỜI GIAN)
# ===============================================================
def render_analytics_tab():
    st.header("📊 Báo cáo Hiệu suất")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        days_back = st.slider("Thời gian (Ngày):", 1, 30, 7)
        selected_dev = st.selectbox("Chọn thiết bị:", DEVICES)
    
    start_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
    
    try:
        response = supabase.table("sensor_data").select("time, Speed, Temp").eq("DevAddr", selected_dev).gte("time", start_date).order("time", desc=False).execute()
        df = pd.DataFrame(response.data)
        
        if df.empty:
            st.warning("Chưa có dữ liệu.")
            return
            
        # FIX LỖI: Dùng pd.to_datetime(..., utc=True) và errors='coerce' để xử lý đa định dạng
        df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
        df = df.dropna(subset=['time']) # Loại bỏ dòng lỗi
        df['time'] = df['time'].dt.tz_convert('Asia/Bangkok').dt.tz_localize(None)
        df.set_index('time', inplace=True)
        
        # ... (Phần UI và Biểu đồ Analytics giữ nguyên) ...
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tốc độ TB", f"{df['Speed'].mean():.1f}")
        k2.metric("Nhiệt độ TB", f"{df['Temp'].mean():.1f}")
        k3.metric("Số lần quá tải", f"{len(df[df['Speed']>150])}")
        k4.metric("Tổng bản ghi", f"{len(df)}")
        
        st.markdown("---")
        st.subheader("🔥 Heatmap Hoạt động")
        df['Hour'] = df.index.hour
        df['Date'] = df.index.date
        heatmap_data = df.groupby(['Date', 'Hour'])['Speed'].mean().unstack(fill_value=0)
        
        fig_heat = px.imshow(heatmap_data, labels=dict(x="Giờ", y="Ngày", color="Tốc độ"), aspect="auto", color_continuous_scale="Viridis")
        st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("🔍 Phân bố bất thường")
        df['Status'] = np.where(df['Speed'] > 150, 'Quá tải', 'Bình thường')
        fig_scat = px.scatter(df, x=df.index, y='Speed', color='Status', color_discrete_map={'Quá tải': 'red', 'Bình thường': 'blue'}, opacity=0.6)
        st.plotly_chart(fig_scat, use_container_width=True)

    except Exception as e:
        st.error(f"Lỗi tải báo cáo: {e}")

# ===============================================================
# MAIN
# ===============================================================
st.title("🏭 STANLEY FACTORY INTELLIGENCE")
st.markdown("---")

tab1, tab2 = st.tabs(["🚀 REAL-TIME MONITOR", "📈 ANALYTICS"])

with tab1:
    render_realtime_tab()
with tab2:
    render_analytics_tab()