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
from datetime import datetime, timedelta
from supabase import create_client

# ===============================================================
# 1. CẤU HÌNH & KẾT NỐI
# ===============================================================
st.set_page_config(page_title="Stanley Factory Monitor - 4 Lanes", layout="wide", page_icon="🏭")

st.markdown("""
<style>
    .status-ok { background-color: #d1e7dd; color: #0f5132; padding: 4px 12px; border-radius: 20px; font-weight: 600; border: 1px solid #badbcc; display: inline-block; }
    .status-err { background-color: #f8d7da; color: #842029; padding: 4px 12px; border-radius: 20px; font-weight: 600; border: 1px solid #f5c2c7; display: inline-block; }
    .status-warn { background-color: #fff3cd; color: #856404; padding: 4px 12px; border-radius: 20px; font-weight: 600; border: 1px solid #ffeeba; display: inline-block; }
    div[data-testid="stMetricValue"] { font-size: 20px !important; color: #333; }
    h3 { font-size: 1.1rem !important; font-weight: 700 !important; color: #444; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #ff4b4b; }
</style>
""", unsafe_allow_html=True)

# Cập nhật đường dẫn model theo Notebook
MODEL_PATH = "saved_models_v2/lstm_factory_v2.pth"
SCALER_PATH = "saved_models_v2/robust_scaler_v2.pkl"
CONFIG_PATH = "saved_models_v2/model_config_v2.pkl"

DEVICES = ["4417930D77DA", "AC0BFBCE8797"]
CHANNELS = ["01", "02"]
REFRESH_RATE = 2 
TEMP_CRASH_THRESHOLD = 40.0

# Lấy Secrets
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except:
    st.error("❌ Thiếu cấu hình Secrets! Vui lòng kiểm tra file .streamlit/secrets.toml")
    st.stop()

@st.cache_resource
def init_connection():
    return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_connection()

# --- LOAD AI MODEL ---
@st.cache_resource
def load_ai():
    # Kiểm tra file tồn tại
    if not os.path.exists(MODEL_PATH):
        # Fallback về thư mục gốc nếu không thấy trong folder v2
        if os.path.exists("lstm_factory_v2.pth"):
            return load_ai_from_path("lstm_factory_v2.pth", "robust_scaler_v2.pkl", "model_config_v2.pkl")
        return None, None, None
    return load_ai_from_path(MODEL_PATH, SCALER_PATH, CONFIG_PATH)

def load_ai_from_path(m_path, s_path, c_path):
    try:
        cfg = joblib.load(c_path)
        scl = joblib.load(s_path)
        
        class LSTMModel(nn.Module):
            def __init__(self, n_features, hidden_dim=128, num_layers=3, dropout=0.2):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(n_features, hidden_dim, num_layers, batch_first=True, dropout=dropout)
                self.fc = nn.Linear(hidden_dim, n_features)
            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return out

        model = LSTMModel(n_features=cfg['n_features'], hidden_dim=cfg['hidden_dim'])
        model.load_state_dict(torch.load(m_path, map_location='cpu'))
        model.eval()
        return model, scl, cfg
    except Exception as e:
        st.error(f"Lỗi load AI: {e}")
        return None, None, None

model, scaler, config = load_ai()

# --- STATE MANAGEMENT (KEY THEO CẶP DEV-CHANNEL) ---
if 'init_done' not in st.session_state:
    # Buffer đếm số lần cảnh báo liên tiếp cho từng làn
    st.session_state.buffer = {(d, c): 0 for d in DEVICES for c in CHANNELS}
    # Log lỗi cho từng làn
    st.session_state.logs = {(d, c): [] for d in DEVICES for c in CHANNELS}
    st.session_state.init_done = True

# --- HÀM HỖ TRỢ DỮ LIỆU ---
def get_recent_data(limit=1000): 
    try:
        response = supabase.table("sensor_data").select("*").order("time", desc=True).limit(limit).execute()
        df = pd.DataFrame(response.data)
        if not df.empty:
            df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
            df['time'] = df['time'].dt.tz_convert('Asia/Bangkok').dt.tz_localize(None)
            df = df.sort_values('time')
        return df
    except: return pd.DataFrame()

# --- AI LOGIC (PER LANE) ---
def predict_anomaly(df_lane, model, scaler, config):
    SEQ_LEN = 30
    if len(df_lane) < SEQ_LEN + 1: return 0.0, False
    
    features = config['features_list']
    try:
        # Lấy đúng features cần thiết
        data_segment = df_lane[features].tail(SEQ_LEN + 1).values
        # Log Transform (Quan trọng: Phải khớp với notebook)
        data_log = np.log1p(data_segment)
        # Scaling
        data_scaled = scaler.transform(data_log)
        
        X_input = data_scaled[:-1]
        Y_actual = data_scaled[-1]
        X_tensor = torch.tensor(X_input, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            Y_pred = model(X_tensor).numpy()[0]
            
        target_idx = config.get('target_cols_idx', [0, 1, 2])
        loss = np.mean(np.abs(Y_pred[target_idx] - Y_actual[target_idx]))
        
        # Ngưỡng từ Config
        is_anomaly = loss > config['threshold']
        return loss, is_anomaly
    except Exception: 
        return 0.0, False

def determine_status_lane(df_lane, model, scaler, config):
    if df_lane.empty or len(df_lane) < 2:
        return 0.0, False, "gray", "NO DATA", "Chưa có dữ liệu"

    last_row = df_lane.iloc[-1]
    prev_row = df_lane.iloc[-2]
    
    # Check kết nối
    time_diff = (last_row['time'] - prev_row['time']).total_seconds()
    if time_diff > 120: # Cho phép delay tới 2 phút
        return 0.0, False, "orange", "⚠️ MẤT KẾT NỐI", f"Mất tin hiệu {int(time_diff)}s"

    speed = last_row['Speed']
    temp = last_row.get('Temp', 0)

    # Check trạng thái dừng
    if speed == 0:
        if temp > TEMP_CRASH_THRESHOLD:
            return 9.99, True, "red", "⛔ CRASH", f"Dừng gấp! Nhiệt cao: {temp}°C"
        return 0.0, False, "gray", "💤 IDLE", "Máy đang tạm nghỉ"

    # Check AI
    if model and scaler:
        loss, is_anomaly = predict_anomaly(df_lane, model, scaler, config)
        if is_anomaly:
            status = "🐢 JAM/SLOW" if speed < 1.5 else "⚠️ OVERLOAD"
            color = "orange" if speed < 1.5 else "red"
            msg = f"Bất thường (Loss: {loss:.2f})"
            return loss, True, color, status, msg
        return loss, False, "green", "✅ RUNNING", "Hoạt động tốt"
            
    return 0.0, False, "gray", "LOADING", "Đang tải AI..."

# --- UI COMPONENTS ---
def create_gauge(value, title, max_val=5, color="green"):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = value,
        title = {'text': title, 'font': {'size': 14, 'color': '#555'}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 1}, 
            'bar': {'color': color},
            'bgcolor': "white", 'borderwidth': 1, 'bordercolor': "#eee",
            'steps': [{'range': [0, max_val*0.3], 'color': '#f0fff4'}, {'range': [max_val*0.3, max_val*0.7], 'color': '#dcfce7'}],
        }
    ))
    fig.update_layout(height=160, margin=dict(t=30,b=10,l=20,r=20))
    return fig

def create_trend_chart_lane(df, title):
    fig = go.Figure()
    if not df.empty:
        latest_time = df['time'].max()
        window_start = latest_time - timedelta(minutes=30)
        df_view = df[df['time'] >= window_start]
        
        fig.add_trace(go.Scatter(x=df_view['time'], y=df_view['Speed'], fill='tozeroy', mode='lines', line=dict(width=2, color='#0ea5e9'), name='Tốc độ'))
        if 'Temp' in df_view.columns:
            fig.add_trace(go.Scatter(x=df_view['time'], y=df_view['Temp'], mode='lines', line=dict(color='#f97316', dash='dot', width=1), yaxis='y2', name='Nhiệt độ'))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        height=200, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showgrid=False, tickformat='%H:%M'),
        yaxis=dict(range=[0, 5], showgrid=True),
        yaxis2=dict(overlaying='y', side='right', showgrid=False, range=[0, 100]),
        legend=dict(orientation="h", y=1.1)
    )
    return fig

def render_lane_ui(dev, ch, df_lane):
    now_str = datetime.now().strftime('%H:%M:%S')
    
    if df_lane.empty:
        st.warning(f"Làn {ch}: Chưa có dữ liệu.")
        return

    last = df_lane.iloc[-1]
    score, is_danger, color_code, status_text, log_msg = determine_status_lane(df_lane, model, scaler, config)

    # Logic Buffer: Phải có 2 lần báo lỗi liên tiếp mới tính là lỗi (tránh nhiễu)
    key = (dev, ch)
    if key not in st.session_state.buffer: st.session_state.buffer[key] = 0
    if key not in st.session_state.logs: st.session_state.logs[key] = []

    if is_danger: st.session_state.buffer[key] += 1
    else: st.session_state.buffer[key] = 0
    
    final_anomaly = (st.session_state.buffer[key] >= 2) or ("CRASH" in status_text)

    # Ghi log nếu có lỗi mới
    if final_anomaly:
        if not st.session_state.logs[key] or st.session_state.logs[key][-1]['msg'] != log_msg:
            st.session_state.logs[key].append({'time': last['time'], 'msg': log_msg})

    # Render Card
    css = "status-ok" if color_code == "green" else ("status-err" if color_code == "red" else "status-warn")
    gauge_col = "#10b981" if color_code == "green" else ("#ef4444" if color_code == "red" else "#f59e0b")

    with st.container(border=True):
        # Header Làn
        c1, c2 = st.columns([1.5, 1])
        c1.markdown(f"#### 🛤️ Làn {ch}")
        c2.markdown(f'<div class="{css}">{status_text}</div>', unsafe_allow_html=True)
        
        # Gauge & Metrics
        g_col, m_col = st.columns([1, 1.2])
        with g_col:
            st.plotly_chart(create_gauge(last['Speed'], "Tốc độ (sp/20s)", 5, gauge_col), use_container_width=True, key=f"g_{dev}_{ch}_{now_str}")
        with m_col:
            st.markdown(f"**Sản lượng:** `{int(last['Actual']):,}`")
            st.markdown(f"**Runtime:** `{int(last.get('RunTime',0)/60)}p`")
            st.markdown(f"**Nhiệt độ:** `{last.get('Temp',0):.1f}°C`")
            st.markdown(f"**AI Loss:** `{score:.3f}`")
        
        # Biểu đồ xu hướng nhỏ
        st.plotly_chart(create_trend_chart_lane(df_lane, "Xu hướng 30p gần nhất"), use_container_width=True, key=f"t_{dev}_{ch}_{now_str}")

        # Logs
        with st.expander("📜 Nhật ký sự cố", expanded=final_anomaly):
            if st.session_state.logs[key]:
                log_df = pd.DataFrame(st.session_state.logs[key])
                log_df['time'] = log_df['time'].dt.strftime('%H:%M:%S')
                st.dataframe(log_df.iloc[::-1].head(5), hide_index=True, use_container_width=True)
            else: st.caption("Hệ thống ổn định.")

# ===============================================================
# MAIN APP
# ===============================================================
st.title("🏭 STANLEY INTELLIGENCE - 4 LANES MONITOR")

# TẠO 3 TAB CHÍNH
tab_names = [f"🏗️ MÁY 1\n({DEVICES[0][-4:]})", f"🏗️ MÁY 2\n({DEVICES[1][-4:]})", "📊 ANALYTICS"]
tab_m1, tab_m2, tab_analytics = st.tabs(tab_names)

# Fetch dữ liệu 1 lần cho hiệu quả
df_all = get_recent_data(600)

# --- TAB MÁY 1 ---
with tab_m1:
    st.subheader(f"Thiết bị: {DEVICES[0]}")
    if not df_all.empty:
        col_l1, col_l2 = st.columns(2)
        dev = DEVICES[0]
        # Lọc dữ liệu theo Channel (Quan trọng!)
        df_ch1 = df_all[(df_all['DevAddr']==dev) & (df_all['Channel']=="01")].sort_values('time')
        df_ch2 = df_all[(df_all['DevAddr']==dev) & (df_all['Channel']=="02")].sort_values('time')
        
        with col_l1: render_lane_ui(dev, "01", df_ch1)
        with col_l2: render_lane_ui(dev, "02", df_ch2)
    else: st.info("⏳ Đang tải dữ liệu...")

# --- TAB MÁY 2 ---
with tab_m2:
    st.subheader(f"Thiết bị: {DEVICES[1]}")
    if not df_all.empty:
        col_l1, col_l2 = st.columns(2)
        dev = DEVICES[1]
        df_ch1 = df_all[(df_all['DevAddr']==dev) & (df_all['Channel']=="01")].sort_values('time')
        df_ch2 = df_all[(df_all['DevAddr']==dev) & (df_all['Channel']=="02")].sort_values('time')
        
        with col_l1: render_lane_ui(dev, "01", df_ch1)
        with col_l2: render_lane_ui(dev, "02", df_ch2)
    else: st.info("⏳ Đang tải dữ liệu...")

# --- TAB ANALYTICS ---
with tab_analytics:
    st.header("📊 Phân tích Hiệu suất Làn")
    
    # Selector thông minh chọn Làn cụ thể
    col_sel1, col_sel2 = st.columns([1, 3])
    with col_sel1:
        selected_option = st.selectbox("Chọn Làn để phân tích:", 
                                   [f"{d} - Kênh {c}" for d in DEVICES for c in CHANNELS])
        days_back = st.slider("Thời gian (Ngày):", 1, 30, 7)
        if st.button("Tải báo cáo"): st.rerun()

    # Parse lựa chọn
    sel_dev = selected_option.split(" - Kênh ")[0]
    sel_ch = selected_option.split(" - Kênh ")[1]

    # Load dữ liệu lịch sử
    start_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
    try:
        # Query Supabase có filter Channel
        response = supabase.table("sensor_data")\
            .select("time, Speed, Temp, Actual, Channel")\
            .eq("DevAddr", sel_dev)\
            .eq("Channel", sel_ch)\
            .gte("time", start_date)\
            .order("time", desc=False)\
            .execute()
            
        df_hist = pd.DataFrame(response.data)
        
        if not df_hist.empty:
            df_hist['time'] = pd.to_datetime(df_hist['time'], format='mixed', utc=True)
            df_hist['time'] = df_hist['time'].dt.tz_convert('Asia/Bangkok').dt.tz_localize(None)
            
            # KPI
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Tốc độ TB", f"{df_hist['Speed'].mean():.2f}")
            k2.metric("Sản lượng Tổng", f"{df_hist['Actual'].max() - df_hist['Actual'].min():,}")
            k3.metric("Nhiệt độ TB", f"{df_hist['Temp'].mean():.1f} °C")
            k4.metric("Số bản ghi", f"{len(df_hist)}")
            
            st.markdown("---")
            
            # Biểu đồ
            fig = px.line(df_hist, x='time', y='Speed', title=f"Biểu đồ Tốc độ - Làn {sel_ch} ({sel_dev[-4:]})")
            st.plotly_chart(fig, use_container_width=True)
            
            # Pie Chart trạng thái
            df_hist['State'] = np.where(df_hist['Speed'] > 0, 'Running', 'Idle')
            fig_pie = px.pie(df_hist, names='State', title="Tỷ lệ Vận hành")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        else:
            st.warning("Không tìm thấy dữ liệu lịch sử cho làn này.")
            
    except Exception as e:
        st.error(f"Lỗi tải Analytics: {e}")

# Auto Refresh logic
time.sleep(REFRESH_RATE)
st.rerun()