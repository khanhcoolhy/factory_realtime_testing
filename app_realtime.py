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
from datetime import datetime, timedelta
from supabase import create_client

# ===============================================================
# 1. CẤU HÌNH & KẾT NỐI
# ===============================================================
st.set_page_config(page_title="Stanley Factory Monitor", layout="wide", page_icon="🏭")

st.markdown("""
<style>
    .status-ok { background-color: #d1e7dd; color: #0f5132; padding: 4px 12px; border-radius: 20px; font-weight: 600; border: 1px solid #badbcc; display: inline-block; }
    .status-err { background-color: #f8d7da; color: #842029; padding: 4px 12px; border-radius: 20px; font-weight: 600; border: 1px solid #f5c2c7; display: inline-block; }
    .css-1r6slb0 { border: 1px solid #e0e0e0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #333; }
    h3 { font-size: 1.2rem !important; font-weight: 700 !important; color: #444; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "lstm_factory_v2.pth"
SCALER_PATH = "robust_scaler_v2.pkl"
CONFIG_PATH = "model_config_v2.pkl"
DEVICES = ["4417930D77DA", "AC0BFBCE8797"]
REFRESH_RATE = 2  # Refresh nhanh (2s) để thấy hiệu ứng trôi

try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
except:
    st.error("❌ Thiếu cấu hình Secrets!")
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

def get_recent_data(limit=1000): # Lấy nhiều data để vẽ đủ lịch sử
    try:
        response = supabase.table("sensor_data").select("*").order("time", desc=True).limit(limit).execute()
        df = pd.DataFrame(response.data)
        if not df.empty:
            # Xử lý thời gian chuẩn xác
            df['time'] = pd.to_datetime(df['time'], utc=True)
            df['time'] = df['time'].dt.tz_convert('Asia/Bangkok').dt.tz_localize(None)
            
            # Lấy data trong 12h qua để đảm bảo liền mạch
            cutoff_time = datetime.now() - timedelta(hours=12)
            df = df[df['time'] > cutoff_time]
            
        return df
    except: return pd.DataFrame()

# ===============================================================
# 2. UI COMPONENTS (GAUGE & CHART)
# ===============================================================

def create_gauge(value, title, max_val=300, color="green"):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = value,
        title = {'text': title, 'font': {'size': 18, 'color': '#555'}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white", 'borderwidth': 1, 'bordercolor': "#ddd",
            'steps': [{'range': [0, max_val*0.3], 'color': '#f0fff4'}, {'range': [max_val*0.3, max_val*0.7], 'color': '#dcfce7'}, {'range': [max_val*0.7, max_val], 'color': '#bbf7d0'}],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': max_val * 0.9}
        }
    ))
    fig.update_layout(height=200, margin=dict(t=40,b=10,l=25,r=25))
    return fig

# --- LOGIC BIỂU ĐỒ: GROWING -> SCROLLING ---
def create_trend_chart(df, dev_name):
    fig = go.Figure()
    
    # Cấu hình trục X mặc định (Growing Mode)
    xaxis_config = dict(
        showgrid=False, 
        tickformat='%H:%M:%S',
    )
    
    title_text = "Lịch sử vận hành"

    if not df.empty:
        # 1. Xác định thời lượng dữ liệu đang có
        min_time = df['time'].min()
        max_time = df['time'].max()
        duration = (max_time - min_time).total_seconds() / 60 # phút
        
        # 2. Chọn chế độ hiển thị
        if duration < 15:
            # CHẾ ĐỘ GROWING: Ít hơn 15p -> Để tự động co giãn (Auto Range)
            # Biểu đồ sẽ "dài ra" từ từ theo dữ liệu
            title_text = f"Lịch sử vận hành (Tích lũy {int(duration)} phút)"
        else:
            # CHẾ ĐỘ SCROLLING: Nhiều hơn 15p -> Khóa khung nhìn 15p cuối
            # Biểu đồ sẽ "trôi"
            now_vn = datetime.utcnow() + timedelta(hours=7)
            xaxis_config.update(dict(
                range=[now_vn - timedelta(minutes=15), now_vn], # Khóa khung 15p
                fixedrange=True
            ))
            title_text = "Lịch sử vận hành (Live Scroll - 15p gần nhất)"

        # Vẽ đường
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['Speed'],
            fill='tozeroy', mode='lines', # Dùng line cho mượt
            line=dict(width=2, color='#0ea5e9'),
            fillcolor='rgba(14, 165, 233, 0.1)',
            name='Tốc độ'
        ))
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['Temp'],
            mode='lines', line=dict(color='#f97316', dash='dot', width=2),
            yaxis='y2', name='Nhiệt độ'
        ))
    
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14, color="#555")),
        height=250, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=xaxis_config,
        yaxis=dict(title="Speed", showgrid=True, gridcolor='#f0f0f0', range=[0, 350]),
        yaxis2=dict(title="Temp (°C)", overlaying='y', side='right', showgrid=False, range=[0, 60]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='white', hovermode="x unified"
    )
    return fig

# ===============================================================
# 3. REAL-TIME TAB LOGIC
# ===============================================================
def render_realtime_tab():
    now_str = (datetime.utcnow() + timedelta(hours=7)).strftime('%H:%M:%S')
    st.caption(f"Last update: {now_str} | Mode: Auto-Smart Chart")
    
    @st.fragment(run_every=REFRESH_RATE)
    def update_loop():
        # Lấy đủ data để vẽ lịch sử dài
        df_all = get_recent_data(1000)
        
        col1, col2 = st.columns(2)
        cols_map = {DEVICES[0]: col1, DEVICES[1]: col2}

        if df_all.empty:
            for dev in DEVICES:
                with cols_map[dev]: st.warning("⏳ Đang chờ Worker bơm dữ liệu...")
            return

        for dev in DEVICES:
            df = df_all[df_all['DevAddr'] == dev].sort_values('time')
            if df.empty: 
                with cols_map[dev]: st.info("Chưa có dữ liệu.")
                continue
            
            last = df.iloc[-1]
            current_col = cols_map[dev]
            
            # AI Logic (Rút gọn)
            score = 0.0; is_danger = False
            if st.session_state.status[dev]: gauge_color = "#ef4444"
            else: gauge_color = "#10b981"

            with current_col:
                with st.container(border=True):
                    h1, h2 = st.columns([3, 1])
                    h1.subheader(f"📡 Device: {dev[-4:]}")
                    
                    if st.session_state.status[dev]: h2.markdown(f'<div class="status-err">⚠️ ERROR</div>', unsafe_allow_html=True)
                    else: h2.markdown(f'<div class="status-ok">✅ RUNNING</div>', unsafe_allow_html=True)

                    st.markdown("---")
                    g1, g2 = st.columns(2)
                    g1.plotly_chart(create_gauge(last['Speed'], "Tốc độ (sp/p)", 300, gauge_color), use_container_width=True, key=f"g_s_{dev}")
                    g2.plotly_chart(create_gauge(last['Temp'], "Nhiệt độ (°C)", 100, "#f59e0b"), use_container_width=True, key=f"g_t_{dev}")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Sản lượng", f"{last['Actual']:,}")
                    m2.metric("Thời gian chạy", f"{int(last['RunTime']/60)}m")
                    m3.metric("AI Score", f"{score:.2f}")

                    st.markdown("---")
                    # GỌI BIỂU ĐỒ THÔNG MINH (GROWING -> SCROLLING)
                    fig_trend = create_trend_chart(df, dev)
                    st.plotly_chart(fig_trend, use_container_width=True, key=f"trend_{dev}")

                    with st.expander("📝 Nhật ký cảnh báo", expanded=False):
                        if st.session_state.logs[dev]:
                            st.dataframe(pd.DataFrame(st.session_state.logs[dev]).head(5), hide_index=True, use_container_width=True)
                        else:
                            st.info("Hệ thống hoạt động ổn định.")

    update_loop()

# ===============================================================
# 4. REPORT TAB (DONUT CHART + HISTOGRAM)
# ===============================================================
def render_analytics_tab():
    st.header("📊 Báo cáo Hiệu suất & OEE")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        days_back = st.slider("Thời gian (Ngày):", 1, 30, 7)
        selected_dev = st.selectbox("Chọn thiết bị:", DEVICES)
    
    start_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
    try:
        response = supabase.table("sensor_data").select("time, Speed, Temp").eq("DevAddr", selected_dev).gte("time", start_date).order("time", desc=False).execute()
        df = pd.DataFrame(response.data)
        
        if df.empty:
            st.warning("Chưa có dữ liệu để phân tích.")
            return
            
        df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert('Asia/Bangkok').dt.tz_localize(None)
        df.set_index('time', inplace=True)

        # KPI
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tốc độ TB", f"{df['Speed'].mean():.1f}")
        k2.metric("Tốc độ Max", f"{df['Speed'].max():.0f}")
        k3.metric("Nhiệt độ TB", f"{df['Temp'].mean():.1f} °C")
        k4.metric("Tổng bản ghi", f"{len(df)}")
        
        st.markdown("---")

        # 1. BIỂU ĐỒ DONUT (Trạng thái)
        conditions = [(df['Speed'] == 0), (df['Speed'] > 0) & (df['Speed'] <= 150), (df['Speed'] > 150)]
        choices = ['Dừng (Idle)', 'Hoạt động (Running)', 'Quá tải (Overload)']
        df['State'] = np.select(conditions, choices, default='Không rõ')

        st.subheader("⏱️ Tỷ lệ Thời gian Vận hành")
        state_counts = df['State'].value_counts().reset_index()
        state_counts.columns = ['State', 'Count']
        
        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(state_counts, values='Count', names='State', hole=0.4, color='State',
                             color_discrete_map={'Dừng (Idle)': '#9e9e9e', 'Hoạt động (Running)': '#2ecc71', 'Quá tải (Overload)': '#e74c3c'})
            fig_pie.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.dataframe(state_counts, use_container_width=True, hide_index=True)

        # 2. BIỂU ĐỒ HISTOGRAM (Phân bố tốc độ)
        st.subheader("📊 Phân bố Tốc độ")
        fig_hist = px.histogram(df, x="Speed", nbins=30, color_discrete_sequence=['#3498db'])
        fig_hist.update_layout(height=300, bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)

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