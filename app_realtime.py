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
st.set_page_config(page_title="Stanley Factory Monitor - Dual View", layout="wide", page_icon="🏭")

# Custom CSS để giao diện đẹp hơn, chia cột rõ ràng
st.markdown("""
<style>
    .status-ok { background-color: #d1e7dd; color: #0f5132; padding: 4px 12px; border-radius: 20px; font-weight: 600; border: 1px solid #badbcc; display: inline-block; }
    .status-err { background-color: #f8d7da; color: #842029; padding: 4px 12px; border-radius: 20px; font-weight: 600; border: 1px solid #f5c2c7; display: inline-block; }
    .status-warn { background-color: #fff3cd; color: #856404; padding: 4px 12px; border-radius: 20px; font-weight: 600; border: 1px solid #ffeeba; display: inline-block; }
    div[data-testid="stMetricValue"] { font-size: 20px !important; color: #333; }
    h3 { font-size: 1.1rem !important; font-weight: 700 !important; color: #444; }
    
    /* Tùy chỉnh Tab cho to và dễ bấm */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 5px; padding: 0 20px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #e6f3ff; border: 2px solid #0ea5e9; color: #0ea5e9; }
</style>
""", unsafe_allow_html=True)

# --- CONFIG ---
MODEL_PATH = "saved_models_v2/lstm_factory_v2.pth"
SCALER_PATH = "saved_models_v2/robust_scaler_v2.pkl"
CONFIG_PATH = "saved_models_v2/model_config_v2.pkl"

DEVICES = ["4417930D77DA", "AC0BFBCE8797"] # Máy 1, Máy 2
CHANNELS = ["01", "02"] # Làn 1, Làn 2
REFRESH_RATE = 2 
TEMP_CRASH_THRESHOLD = 40.0

# --- SUPABASE ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except:
    st.error("❌ Thiếu cấu hình Secrets!")
    st.stop()

@st.cache_resource
def init_connection():
    return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_connection()

# --- LOAD AI ---
@st.cache_resource
def load_ai():
    # Fallback đường dẫn
    if not os.path.exists(MODEL_PATH):
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
    except: return None, None, None

model, scaler, config = load_ai()

# --- STATE ---
if 'init_done' not in st.session_state:
    st.session_state.buffer = {(d, c): 0 for d in DEVICES for c in CHANNELS}
    st.session_state.logs = {(d, c): [] for d in DEVICES for c in CHANNELS}
    st.session_state.init_done = True

# --- HELPER FUNCTIONS ---
def get_recent_data(limit=800): 
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
    SEQ_LEN = 30
    if len(df_lane) < SEQ_LEN + 1: return 0.0, False
    try:
        features = config['features_list']
        data_segment = df_lane[features].tail(SEQ_LEN + 1).values
        data_log = np.log1p(data_segment)
        data_scaled = scaler.transform(data_log)
        
        X_input = torch.tensor(data_scaled[:-1], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad(): Y_pred = model(X_input).numpy()[0]
        Y_actual = data_scaled[-1]
        
        target_idx = config.get('target_cols_idx', [0, 1, 2])
        loss = np.mean(np.abs(Y_pred[target_idx] - Y_actual[target_idx]))
        return loss, loss > config['threshold']
    except: return 0.0, False

def determine_status(df_lane):
    if df_lane.empty: return 0.0, False, "gray", "NO DATA", "Chưa có dữ liệu"
    last = df_lane.iloc[-1]
    
    # Check Offline
    if (datetime.now() - last['time']).total_seconds() > 120:
        return 0.0, False, "orange", "⚠️ MẤT KẾT NỐI", "Offline > 2 phút"
    
    # Check Dừng
    if last['Speed'] == 0:
        if last.get('Temp', 0) > TEMP_CRASH_THRESHOLD:
            return 9.9, True, "red", "⛔ CRASH", f"Dừng gấp! Nhiệt: {last['Temp']}°C"
        return 0.0, False, "gray", "💤 MÁY NGHỈ", "Đang dừng theo kế hoạch"

    # Check AI
    if model:
        loss, is_anom = predict_anomaly(df_lane, model, scaler, config)
        if is_anom:
            stt = "🐢 CHẠY CHẬM" if last['Speed'] < 1.5 else "⚠️ QUÁ TẢI"
            clr = "orange" if last['Speed'] < 1.5 else "red"
            return loss, True, clr, stt, f"AI Loss cao: {loss:.2f}"
        return loss, False, "green", "✅ ỔN ĐỊNH", "Hoạt động tốt"
    
    return 0.0, False, "gray", "LOADING", "Đang tải AI..."

# --- UI COMPONENTS ---
def create_gauge(val, title, color):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = val,
        title = {'text': title, 'font': {'size': 14}},
        gauge = {'axis': {'range': [None, 5]}, 'bar': {'color': color}}
    ))
    fig.update_layout(height=150, margin=dict(t=30,b=10,l=20,r=20))
    return fig

def render_lane_card(dev, ch, df_lane):
    """Hàm vẽ giao diện cho 1 Làn (Card)"""
    now_str = datetime.now().strftime('%H:%M:%S')
    
    if df_lane.empty:
        st.info(f"Làn {ch}: Chưa có dữ liệu")
        return

    last = df_lane.iloc[-1]
    score, is_danger, color, status_text, log_msg = determine_status(df_lane)

    # Logic Buffer Alert
    key = (dev, ch)
    if is_danger: st.session_state.buffer[key] += 1
    else: st.session_state.buffer[key] = 0
    final_alert = (st.session_state.buffer[key] >= 2) or ("CRASH" in status_text)

    # Log
    if final_alert:
        if not st.session_state.logs[key] or st.session_state.logs[key][-1]['msg'] != log_msg:
            st.session_state.logs[key].append({'time': last['time'], 'msg': log_msg})

    # --- UI CARD ---
    css = "status-ok" if color == "green" else ("status-err" if color == "red" else ("status-warn" if color == "orange" else "status-gray"))
    gauge_col = "#10b981" if color == "green" else ("#ef4444" if color == "red" else "#f59e0b")

    with st.container(border=True):
        # Header: Tên Làn + Trạng thái
        c1, c2 = st.columns([1, 1])
        c1.markdown(f"#### 🛣️ Làn {ch}")
        c2.markdown(f'<div class="{css}" style="float:right">{status_text}</div>', unsafe_allow_html=True)
        
        # Đồng hồ + Chỉ số
        gc, mc = st.columns([1, 1.2])
        with gc:
            st.plotly_chart(create_gauge(last['Speed'], "Tốc độ (sp/20s)", gauge_col), use_container_width=True, key=f"g_{dev}_{ch}_{now_str}")
        with mc:
            st.markdown(f"📦 **Sản lượng:** `{int(last['Actual']):,}`")
            st.markdown(f"⏱️ **Runtime:** `{int(last.get('RunTime',0)/60)}p`")
            st.markdown(f"🌡️ **Nhiệt độ:** `{last.get('Temp',0):.1f}°C`")
            st.markdown(f"🧠 **AI Loss:** `{score:.3f}`")

        # Biểu đồ nhỏ
        chart_data = df_lane.tail(50)
        fig = px.line(chart_data, x='time', y='Speed', height=180)
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, range=[0, 5]))
        st.plotly_chart(fig, use_container_width=True, key=f"c_{dev}_{ch}_{now_str}")

        # Logs expander
        with st.expander("📜 Lịch sử báo lỗi", expanded=final_alert):
            if st.session_state.logs[key]:
                l_df = pd.DataFrame(st.session_state.logs[key])
                l_df['time'] = l_df['time'].dt.strftime('%H:%M:%S')
                st.dataframe(l_df.iloc[::-1].head(5), hide_index=True, use_container_width=True)
            else: st.caption("Hệ thống ổn định.")

# ===============================================================
# MAIN APP LAYOUT (3 TABS)
# ===============================================================
st.title("🏭 STANLEY FACTORY INTELLIGENCE")

# Định nghĩa 3 Tab
tab1, tab2, tab3 = st.tabs([
    f"🏗️ MÁY 1 (Đuôi 77DA)", 
    f"🏗️ MÁY 2 (Đuôi 8797)", 
    "📊 ANALYTICS"
])

# Lấy dữ liệu 1 lần
df_all = get_recent_data(600)

# --- TAB 1: MÁY 1 (Hiển thị 2 làn song song) ---
with tab1:
    st.markdown("### 📡 Trạng thái Máy 1 (4417930D77DA)")
    if not df_all.empty:
        # Chia làm 2 cột: Trái (Làn 01) - Phải (Làn 02)
        col_left, col_right = st.columns(2)
        
        dev_id = DEVICES[0] # Máy 1
        
        # Lọc dữ liệu cho từng làn
        df_lane1 = df_all[(df_all['DevAddr'] == dev_id) & (df_all['Channel'] == "01")].sort_values('time')
        df_lane2 = df_all[(df_all['DevAddr'] == dev_id) & (df_all['Channel'] == "02")].sort_values('time')
        
        with col_left:
            render_lane_card(dev_id, "01", df_lane1)
        
        with col_right:
            render_lane_card(dev_id, "02", df_lane2)
    else:
        st.info("⏳ Đang tải dữ liệu Máy 1...")

# --- TAB 2: MÁY 2 (Hiển thị 2 làn song song) ---
with tab2:
    st.markdown("### 📡 Trạng thái Máy 2 (AC0BFBCE8797)")
    if not df_all.empty:
        # Chia làm 2 cột: Trái (Làn 01) - Phải (Làn 02)
        col_left, col_right = st.columns(2)
        
        dev_id = DEVICES[1] # Máy 2
        
        # Lọc dữ liệu
        df_lane1 = df_all[(df_all['DevAddr'] == dev_id) & (df_all['Channel'] == "01")].sort_values('time')
        df_lane2 = df_all[(df_all['DevAddr'] == dev_id) & (df_all['Channel'] == "02")].sort_values('time')
        
        with col_left:
            render_lane_card(dev_id, "01", df_lane1)
        
        with col_right:
            render_lane_card(dev_id, "02", df_lane2)
    else:
        st.info("⏳ Đang tải dữ liệu Máy 2...")

# --- TAB 3: ANALYTICS ---
with tab3:
    st.markdown("### 📊 Phân tích hiệu suất & Dự báo")
    
    c1, c2 = st.columns([1, 3])
    with c1:
        # Selector chọn cụ thể Làn nào để phân tích
        opt = st.selectbox("Chọn Làn:", [f"{d[-4:]} - Làn {c}" for d in DEVICES for c in CHANNELS])
        days = st.slider("Xem lại (ngày):", 1, 30, 7)
        btn = st.button("Tải báo cáo")

    if btn:
        sel_dev_suffix = opt.split(" - ")[0]
        sel_ch = opt.split(" Làn ")[1]
        # Map lại ID đầy đủ
        real_dev_id = DEVICES[0] if DEVICES[0].endswith(sel_dev_suffix) else DEVICES[1]
        
        # Query
        start = (datetime.utcnow() - timedelta(days=days)).isoformat()
        res = supabase.table("sensor_data").select("time,Speed,Temp,Actual").eq("DevAddr", real_dev_id).eq("Channel", sel_ch).gte("time", start).order("time").execute()
        df_his = pd.DataFrame(res.data)
        
        if not df_his.empty:
            df_his['time'] = pd.to_datetime(df_his['time']).dt.tz_convert('Asia/Bangkok').dt.tz_localize(None)
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Tốc độ TB", f"{df_his['Speed'].mean():.2f}")
            k2.metric("Max Speed", f"{df_his['Speed'].max()}")
            k3.metric("Tổng sản lượng", f"{df_his['Actual'].max() - df_his['Actual'].min():,}")
            
            st.plotly_chart(px.line(df_his, x='time', y='Speed', title=f"Biểu đồ Tốc độ: {opt}"), use_container_width=True)
        else:
            st.warning("Không có dữ liệu lịch sử.")

# Refresh tự động
time.sleep(REFRESH_RATE)
st.rerun()