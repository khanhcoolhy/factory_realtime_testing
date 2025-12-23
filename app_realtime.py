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
# 1. CẤU HÌNH HỆ THỐNG & KẾT NỐI
# ===============================================================
st.set_page_config(page_title="Hệ thống Giám sát Nhà máy", layout="wide", page_icon="🏭")

# --- CSS: Giao diện Tab trái & Card ---
st.markdown("""
<style>
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
        padding-top: 20px;
    }
    .stRadio > div {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stRadio label {
        font-weight: bold;
        font-size: 16px;
        padding: 10px 5px;
    }
    
    /* Card Styling */
    div[data-testid="stMetricValue"] { font-size: 22px !important; }
    .status-badge {
        padding: 5px 10px; border-radius: 6px; font-weight: bold; text-align: center; color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIG THIẾT BỊ & MODEL ---
# Logic: Máy 1 gồm Làn 1,2. Máy 2 gồm Làn 3,4.
# Giả định: Mỗi máy vật lý gửi lên Channel "01", "02". 
# Nếu máy 2 gửi "03", "04" trong DB, hãy sửa channel_map bên dưới.
MACHINE_CONFIG = {
    "MÁY 1": {
        "id": "4417930D77DA", # Thay ID thực tế của Máy 1
        "name": "MÁY CẮT 01",
        "lanes": [
            {"code": "L1", "name": "Làn 1", "db_channel": "01"},
            {"code": "L2", "name": "Làn 2", "db_channel": "02"}
        ]
    },
    "MÁY 2": {
        "id": "AC0BFBCE8797", # Thay ID thực tế của Máy 2
        "name": "MÁY CẮT 02",
        "lanes": [
            {"code": "L3", "name": "Làn 3", "db_channel": "01"}, # Hoặc "03" tùy DB
            {"code": "L4", "name": "Làn 4", "db_channel": "02"}  # Hoặc "04" tùy DB
        ]
    }
}

MODEL_PATH = "saved_models_v2/lstm_factory_v2.pth"
SCALER_PATH = "saved_models_v2/robust_scaler_v2.pkl"
CONFIG_PATH = "saved_models_v2/model_config_v2.pkl"
REFRESH_RATE = 2 

# --- SUPABASE ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except:
    st.error("❌ Lỗi: Thiếu Secrets SUPABASE")
    st.stop()

@st.cache_resource
def init_connection():
    return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_connection()

# --- LOAD AI MODEL ---
@st.cache_resource
def load_ai():
    # Kiểm tra file tồn tại để tránh crash
    if not os.path.exists(CONFIG_PATH): return None, None, None
    try:
        cfg = joblib.load(CONFIG_PATH)
        scl = joblib.load(SCALER_PATH)
        
        # Định nghĩa lại class Model giống hệt Notebook
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
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        return model, scl, cfg
    except Exception as e:
        print(f"Lỗi load AI: {e}")
        return None, None, None

model, scaler, config = load_ai()

# --- SESSION STATE ---
if 'logs' not in st.session_state:
    st.session_state.logs = [] # Lưu log chung

# ===============================================================
# 2. XỬ LÝ DỮ LIỆU & AI
# ===============================================================
def get_realtime_data(limit=500): 
    """Lấy dữ liệu mới nhất từ DB"""
    try:
        response = supabase.table("sensor_data").select("*").order("time", desc=True).limit(limit).execute()
        df = pd.DataFrame(response.data)
        if not df.empty:
            df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
            df['time'] = df['time'].dt.tz_convert('Asia/Bangkok').dt.tz_localize(None)
            df = df.sort_values('time')
        return df
    except: return pd.DataFrame()

def predict_anomaly(df_lane, model, scaler, config):
    """Logic dự báo giống hệt Notebook: Log1p -> Scale -> LSTM"""
    SEQ_LEN = 30
    if len(df_lane) < SEQ_LEN + 1: return 0.0, False
    try:
        features = config['features_list'] # ['Speed', 'd_RunTime', etc.]
        # Lấy đúng các cột feature mà model cần
        data_segment = df_lane[features].tail(SEQ_LEN + 1).values
        
        # 1. Log Transform
        data_log = np.log1p(data_segment)
        # 2. Scale
        data_scaled = scaler.transform(data_log)
        
        # 3. Predict
        X_input = torch.tensor(data_scaled[:-1], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad(): 
            Y_pred = model(X_input).numpy()[0]
            
        Y_actual = data_scaled[-1]
        
        # 4. Calculate Loss (Chỉ trên các cột Target)
        target_idx = config.get('target_cols_idx', [0, 1, 2]) # Speed, RunTime, HeldTime
        loss = np.mean(np.abs(Y_pred[target_idx] - Y_actual[target_idx]))
        
        return loss, loss > config['threshold']
    except Exception as e: 
        return 0.0, False

def analyze_lane_status(df_lane):
    """Phân tích trạng thái logic nghiệp vụ"""
    if df_lane.empty: return "NODATA", "black", "Chờ dữ liệu", 0.0
    
    last = df_lane.iloc[-1]
    now = datetime.now()
    
    # 1. Check Offline (> 3 phút không có tin)
    if (now - last['time']).total_seconds() > 180:
        return "OFFLINE", "gray", f"Mất kết nối {last['time'].strftime('%H:%M')}", 0.0

    # 2. Check Crash (Nhiệt cao + Dừng máy)
    if last['Speed'] == 0 and last.get('Temp', 0) > 40:
        return "CRASH", "red", f"⚠️ QUÁ NHIỆT {last['Temp']}°C", 9.9

    # 3. Check Idle
    if last['Speed'] < 0.1:
        return "IDLE", "#6c757d", "💤 Máy đang nghỉ", 0.0

    # 4. Check AI Anomaly
    loss, is_anom = predict_anomaly(df_lane, model, scaler, config) if model else (0.0, False)
    
    if is_anom:
        if last['Speed'] < 1.5: return "SLOW", "orange", "🐢 Chạy chậm bất thường", loss
        return "OVERLOAD", "red", "🔥 Quá tải / Bất thường", loss
        
    return "OK", "green", "✅ Ổn định", loss

# ===============================================================
# 3. UI COMPONENTS
# ===============================================================
def draw_gauge(value, title, color_code):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = value,
        title = {'text': title, 'font': {'size': 16}},
        gauge = {
            'axis': {'range': [None, 5]},
            'bar': {'color': color_code},
            'bgcolor': "white",
            'steps': [{'range': [0, 1.5], 'color': '#f8f9fa'}, {'range': [1.5, 5], 'color': '#e9ecef'}]
        }
    ))
    fig.update_layout(height=140, margin=dict(t=30, b=10, l=20, r=20))
    return fig

def render_lane_view(lane_cfg, df_lane):
    """Hiển thị giao diện cho 1 Làn"""
    st_code, color, msg, ai_loss = analyze_lane_status(df_lane)
    
    # Header Card
    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        c1.markdown(f"### 🛣️ {lane_cfg['name']}")
        c2.markdown(f'<div style="background-color:{color};" class="status-badge">{msg}</div>', unsafe_allow_html=True)
        st.divider()
        
        if not df_lane.empty:
            last = df_lane.iloc[-1]
            
            # Row 1: Gauge & Metrics
            kc1, kc2 = st.columns([1, 1])
            with kc1:
                st.plotly_chart(draw_gauge(last['Speed'], "Tốc độ (m/s)", color), use_container_width=True)
            with kc2:
                st.metric("📦 Sản lượng", f"{int(last['Actual']):,}")
                st.metric("🌡️ Nhiệt độ", f"{last.get('Temp', 0):.1f}°C")
                st.metric("🧠 AI Loss", f"{ai_loss:.4f}", delta_color="inverse")

            # Row 2: Sparkline Chart
            chart_df = df_lane.tail(50)
            fig = px.area(chart_df, x='time', y='Speed', height=100)
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
            fig.update_traces(line_color=color, fillcolor=color, fill_opacity=0.1)
            st.plotly_chart(fig, use_container_width=True)
            
            # Log Warning
            if st_code in ["CRASH", "SLOW", "OVERLOAD"]:
                log_entry = f"{datetime.now().strftime('%H:%M:%S')} - {lane_cfg['name']}: {msg}"
                if not st.session_state.logs or st.session_state.logs[-1] != log_entry:
                    st.session_state.logs.append(log_entry)

# ===============================================================
# 4. MAIN LAYOUT
# ===============================================================

# --- A. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2620/2620952.png", width=70) # Factory Icon
    st.title("STANLEY IoT")
    st.caption(f"Update: {datetime.now().strftime('%H:%M:%S')}")
    st.markdown("---")
    
    # 3 TAB CHÍNH
    selected_tab = st.radio(
        "KHU VỰC GIÁM SÁT:", 
        ["🏗️ MÁY 1 (Làn 1-2)", "🏗️ MÁY 2 (Làn 3-4)", "📊 ANALYTICS"],
        captions=["Device 77DA", "Device 8797", "Báo cáo lịch sử"]
    )
    
    st.markdown("---")
    if st.session_state.logs:
        st.warning("🔔 Cảnh báo gần đây:")
        for l in st.session_state.logs[-3:]:
            st.caption(l)

# --- B. XỬ LÝ DỮ LIỆU CHUNG ---
df_all = get_realtime_data(1000)

# --- C. HIỂN THỊ NỘI DUNG THEO TAB ---

# === TAB 1: MÁY 1 ===
if "MÁY 1" in selected_tab:
    cfg = MACHINE_CONFIG["MÁY 1"]
    st.header(f"📡 {cfg['name']} - Realtime Monitor")
    
    col_left, col_right = st.columns(2)
    
    # Filter Data for Device 1
    df_dev = df_all[df_all['DevAddr'] == cfg['id']]
    
    with col_left:
        lane_info = cfg['lanes'][0]
        df_l = df_dev[df_dev['Channel'] == lane_info['db_channel']].sort_values('time')
        render_lane_view(lane_info, df_l)
        
    with col_right:
        lane_info = cfg['lanes'][1]
        df_l = df_dev[df_dev['Channel'] == lane_info['db_channel']].sort_values('time')
        render_lane_view(lane_info, df_l)

# === TAB 2: MÁY 2 ===
elif "MÁY 2" in selected_tab:
    cfg = MACHINE_CONFIG["MÁY 2"]
    st.header(f"📡 {cfg['name']} - Realtime Monitor")
    
    col_left, col_right = st.columns(2)
    
    # Filter Data for Device 2
    df_dev = df_all[df_all['DevAddr'] == cfg['id']]
    
    with col_left:
        lane_info = cfg['lanes'][0] # Làn 3
        df_l = df_dev[df_dev['Channel'] == lane_info['db_channel']].sort_values('time')
        render_lane_view(lane_info, df_l)
        
    with col_right:
        lane_info = cfg['lanes'][1] # Làn 4
        df_l = df_dev[df_dev['Channel'] == lane_info['db_channel']].sort_values('time')
        render_lane_view(lane_info, df_l)

# === TAB 3: ANALYTICS ===
else:
    st.header("📊 Phân Tích & Báo Cáo Hiệu Suất")
    st.markdown("---")
    
    # Analytics Controls
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        m_sel = st.selectbox("Chọn Máy:", ["MÁY 1", "MÁY 2"])
    with c2:
        d_sel = st.slider("Xem lại (ngày):", 1, 30, 7)
    
    if st.button("🔍 Tải dữ liệu lịch sử"):
        sel_id = MACHINE_CONFIG[m_sel]['id']
        start_date = (datetime.utcnow() - timedelta(days=d_sel)).isoformat()
        
        # Query History
        res = supabase.table("sensor_data").select("*")\
            .eq("DevAddr", sel_id).gte("time", start_date).order("time").execute()
        df_hist = pd.DataFrame(res.data)
        
        if not df_hist.empty:
            df_hist['time'] = pd.to_datetime(df_hist['time']).dt.tz_convert('Asia/Bangkok')
            
            # Tách Channel để vẽ biểu đồ so sánh
            fig = px.line(df_hist, x='time', y='Speed', color='Channel', 
                          title=f"Biểu đồ tốc độ: {m_sel}", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Thống kê
            c_stat1, c_stat2 = st.columns(2)
            c_stat1.metric("Tổng sản lượng", f"{int(df_hist['Actual'].max() - df_hist['Actual'].min()):,}")
            c_stat2.metric("Nhiệt độ TB", f"{df_hist['Temp'].mean():.1f}°C")
            
        else:
            st.warning("Không tìm thấy dữ liệu trong khoảng thời gian này.")

# Auto Refresh logic
time.sleep(REFRESH_RATE)
st.rerun()