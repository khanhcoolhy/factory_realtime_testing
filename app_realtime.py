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
# 1. CẤU HÌNH HỆ THỐNG & LOAD MODEL
# ===============================================================
MODEL_PATH = "lstm_factory_v2.pth"
SCALER_PATH = "robust_scaler_v2.pkl"
CONFIG_PATH = "model_config_v2.pkl"
DEVICES = ["4417930D77DA", "AC0BFBCE8797"]
REFRESH_RATE = 5  # Tăng lên 5s để đỡ spam API Cloud

st.set_page_config(page_title="Stanley AI Manager", layout="wide", page_icon="🏭")

# --- LẤY SECRETS TỪ STREAMLIT CLOUD ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
except:
    st.error("❌ Chưa cấu hình Secrets! Vui lòng vào Settings trên Streamlit Cloud.")
    st.stop()

# KẾT NỐI SUPABASE
@st.cache_resource
def init_connection():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_connection()

# CSS
st.markdown("""
<style>
    .metric-card { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    div[data-testid="stMetricValue"] { font-size: 20px; }
</style>
""", unsafe_allow_html=True)

# --- LOAD AI MODEL (GIỮ NGUYÊN) ---
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
    if not TELEGRAM_TOKEN or "..." in TELEGRAM_TOKEN: return
    try: requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=2)
    except: pass

def get_action(speed):
    if speed < 50: return "Kiểm tra nguồn điện / Băng tải"
    if speed > 10000: return "Kiểm tra biến tần / Bộ điều khiển"
    return "Kiểm tra trục động cơ / Bôi trơn"

# HÀM LẤY DATA TỪ SUPABASE (THAY THẾ SQLITE)
def get_recent_data(limit=200):
    try:
        response = supabase.table("sensor_data").select("*").order("time", desc=True).limit(limit).execute()
        df = pd.DataFrame(response.data)
        if not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            # Chuyển múi giờ về VN (Supabase lưu UTC)
            df['time'] = df['time'].dt.tz_convert('Asia/Bangkok').dt.tz_localize(None)
        return df
    except Exception as e:
        st.error(f"Lỗi Supabase: {e}")
        return pd.DataFrame()

# ===============================================================
# 2. LOGIC TAB 1: REAL-TIME (GIỮ NGUYÊN UI)
# ===============================================================
def render_realtime_tab():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"📡 {DEVICES[0]}")
        p_slots1 = [st.empty() for _ in range(5)]
        c1, c2, c3 = st.columns(3)
        p_slots1[0] = c1.empty(); p_slots1[1] = c2.empty(); p_slots1[2] = c3.empty()
        p_slots1[3] = st.empty(); p_slots1[4] = st.empty()

    with col2:
        st.subheader(f"📡 {DEVICES[1]}")
        p_slots2 = [st.empty() for _ in range(5)]
        c1, c2, c3 = st.columns(3)
        p_slots2[0] = c1.empty(); p_slots2[1] = c2.empty(); p_slots2[2] = c3.empty()
        p_slots2[3] = st.empty(); p_slots2[4] = st.empty()

    slots_map = {DEVICES[0]: p_slots1, DEVICES[1]: p_slots2}

    def update_ui(dev, df):
        last = df.iloc[-1]
        slots = slots_map[dev]
        
        is_danger = False
        score = 0.0
        if model and len(df) >= 30:
            cols = ['Speed', 'd_RunTime', 'd_HeldTime', 'Temp', 'Humidity']
            data = scaler.transform(np.log1p(df[cols].tail(30).values))
            with torch.no_grad(): pred = model(torch.tensor(data, dtype=torch.float32).unsqueeze(0))
            score = np.mean(np.abs(data[-1, :3] - pred.numpy()[0, :3]))
            is_danger = score > config['threshold']

        if is_danger: st.session_state.buffer[dev] += 1
        else: st.session_state.buffer[dev] = 0
        confirmed = st.session_state.buffer[dev] >= 3
        curr_stat = st.session_state.status[dev]

        if confirmed and not curr_stat:
            send_telegram(f"🔥 **CẢNH BÁO: {dev}**\nSpeed cao: {last['Speed']:.0f}")
            st.session_state.status[dev] = True
            st.session_state.logs[dev].insert(0, {"Time": last['time'].strftime('%H:%M:%S'), "Vấn đề": "Lỗi AI", "Xử lý": get_action(last['Speed'])})
        elif not is_danger and curr_stat:
            send_telegram(f"✅ **ĐÃ ỔN ĐỊNH: {dev}**")
            st.session_state.status[dev] = False

        slots[0].metric("Speed", f"{last['Speed']:.0f}", delta="Run" if last['Speed']>0 else "Stop")
        slots[1].metric("Temp", f"{last['Temp']:.1f}°C")
        status_lbl = "⚠️ LỖI" if st.session_state.status[dev] else "ỔN ĐỊNH"
        status_col = "normal" if st.session_state.status[dev] else "inverse"
        slots[2].metric("Status", status_lbl, delta=f"Score: {score:.2f}", delta_color=status_col)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['time'], y=df['Speed'], fill='tozeroy', line=dict(color='#00CC96')))
        fig.add_trace(go.Scatter(x=df['time'], y=df['Temp'], yaxis='y2', line=dict(color='orange', dash='dot')))
        bg = "rgba(255,0,0,0.1)" if st.session_state.status[dev] else "white"
        fig.update_layout(height=250, margin=dict(t=10,b=0,l=0,r=0), yaxis=dict(range=[0, 350]), yaxis2=dict(overlaying='y', side='right', range=[0, 60]), showlegend=False, plot_bgcolor=bg)
        slots[3].plotly_chart(fig, use_container_width=True, key=f"chart_{dev}_{time.time()}")

        if st.session_state.logs[dev]: slots[4].dataframe(pd.DataFrame(st.session_state.logs[dev]).head(5), hide_index=True, use_container_width=True)
        else: slots[4].info("✅ Hoạt động tốt")

    # --- THAY ĐỔI: GỌI HÀM SUPABASE ---
    df_all = get_recent_data(200)
    if not df_all.empty:
        for d in DEVICES:
            df_dev = df_all[df_all['DevAddr'] == d].sort_values('time')
            if not df_dev.empty: update_ui(d, df_dev)

# ===============================================================
# 3. TAB 2: ANALYTICS (ADAPTED FOR SUPABASE)
# ===============================================================
def render_analytics_tab():
    st.header("📊 Báo cáo Hiệu suất Vận hành")
    
    col_filter1, col_filter2 = st.columns(2)
    days_back = col_filter1.slider("Xem dữ liệu trong khoảng (ngày):", 1, 30, 7) # Giới hạn 30 ngày cho demo cloud
    selected_dev = col_filter2.selectbox("Chọn thiết bị phân tích:", DEVICES)
    
    # Tính ngày bắt đầu (UTC)
    start_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
    
    try:
        # QUERY 1: Lấy dữ liệu thô (có filter time & device)
        # Lưu ý: Supabase API free có giới hạn rows, nên cẩn thận khi query lớn
        response = supabase.table("sensor_data")\
            .select("time, Speed, Temp")\
            .eq("DevAddr", selected_dev)\
            .gte("time", start_date)\
            .order("time", desc=False)\
            .execute() # Lấy tối đa mặc định (thường là 1000 dòng)
            
        df_trend = pd.DataFrame(response.data)

        if df_trend.empty:
            st.warning("Chưa có đủ dữ liệu lịch sử.")
            return

        # Xử lý Timezone
        df_trend['time'] = pd.to_datetime(df_trend['time']).dt.tz_convert('Asia/Bangkok').dt.tz_localize(None)
        df_trend = df_trend.set_index('time')
        
        # Filter High Load từ df_trend (đỡ phải query lại)
        df_high_load = df_trend[df_trend['Speed'] > 100].copy()

        # KPI
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        avg_speed = df_trend['Speed'].mean()
        max_speed = df_trend['Speed'].max()
        avg_temp = df_trend['Temp'].mean()
        real_error_count = len(df_high_load)

        kpi1.metric("Tốc độ TB", f"{avg_speed:.2f} sp/p")
        kpi2.metric("Tốc độ Max", f"{max_speed:.0f} sp/p")
        kpi3.metric("Nhiệt độ TB", f"{avg_temp:.1f} °C")
        kpi4.metric("Ghi nhận Tải cao", f"{real_error_count} lần", delta_color="off")

        st.markdown("---")

        # BIỂU ĐỒ 1: Xu hướng
        st.subheader("📈 Xu hướng Vận hành Trung bình")
        df_daily = df_trend.resample('D').mean()
        if not df_daily.empty:
            st.plotly_chart(px.line(df_daily, y=['Speed', 'Temp'], markers=True, height=300), use_container_width=True)

        # BIỂU ĐỒ 2: Scatter
        st.markdown("---")
        st.subheader("💓 Nhịp độ Vận hành (Phân tán)")
        plot_df = df_trend.copy()
        plot_df['Type'] = np.where(plot_df['Speed'] > 100, 'Tải cao', 'Ổn định')
        
        fig_pulse = px.scatter(
            plot_df, x=plot_df.index, y='Speed', color='Type',
            color_discrete_map={'Tải cao': '#ff9800', 'Ổn định': '#4caf50'},
            title="Phân bố các điểm hoạt động", labels={'time': 'Thời gian', 'Speed': 'Tốc độ'}
        )
        fig_pulse.update_traces(marker=dict(size=6, opacity=0.7))
        st.plotly_chart(fig_pulse, use_container_width=True)

        # BIỂU ĐỒ 3: Bar Chart
        st.markdown("---")
        st.subheader(f"⚡ Chi tiết tần suất Tải cao ({real_error_count} lần)")
        
        if not df_high_load.empty:
            daily_counts = df_high_load.resample('D').count()['Speed']
            daily_counts = daily_counts[daily_counts > 0] 
            
            chart_data = daily_counts.reset_index()
            chart_data.columns = ['Date', 'Count']
            chart_data['DateStr'] = chart_data['Date'].dt.strftime('%d/%m')

            fig_bar = px.bar(
                chart_data, x='DateStr', y='Count', text='Count',
                title="Số lần ghi nhận tải cao theo ngày", labels={'DateStr': 'Ngày', 'Count': 'Số lần'},
                color_discrete_sequence=['#ff9800']
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.success("Không có ghi nhận tải cao nào.")
            
    except Exception as e:
        st.error(f"Lỗi tải báo cáo: {e}")

# ===============================================================
# 4. MAIN
# ===============================================================
st.title("🏭 Stanley Smart Factory Monitor")
st.markdown("---")

tab_realtime, tab_report = st.tabs(["🔴 GIÁM SÁT REAL-TIME", "📊 BÁO CÁO & XU HƯỚNG"])

with tab_realtime:
    @st.fragment(run_every=REFRESH_RATE)
    def run_rt(): render_realtime_tab()
    run_rt()

with tab_report:
    render_analytics_tab()