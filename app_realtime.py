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
# 1. CẤU HÌNH GIAO DIỆN & KẾT NỐI
# ===============================================================
# layout="wide" là yếu tố quan trọng nhất để giao diện trải ngang
st.set_page_config(page_title="Stanley Factory Monitor", layout="wide", page_icon="🏭")

# --- CSS LÀM ĐẸP TAB NGANG ---
st.markdown("""
<style>
    /* 1. Tùy chỉnh thanh Tab nằm ngang cho to và rõ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 8px;
        color: #495057;
        font-weight: 600;
        font-size: 16px; /* Chữ to dễ đọc */
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        flex-grow: 1; /* Tự động giãn đều chiều ngang */
        justify-content: center;
    }

    /* Khi Tab được chọn */
    .stTabs [aria-selected="true"] {
        background-color: #007bff !important;
        color: #ffffff !important;
        box-shadow: 0 4px 6px rgba(0,123,255,0.3);
    }

    /* 2. Style cho Card (Khung Làn) */
    .status-ok { background-color: #d1e7dd; color: #0f5132; padding: 4px 12px; border-radius: 12px; font-weight: 700; border: 1px solid #badbcc; }
    .status-err { background-color: #f8d7da; color: #842029; padding: 4px 12px; border-radius: 12px; font-weight: 700; border: 1px solid #f5c2c7; }
    .status-warn { background-color: #fff3cd; color: #856404; padding: 4px 12px; border-radius: 12px; font-weight: 700; border: 1px solid #ffeeba; }
    .status-gray { background-color: #e2e3e5; color: #41464b; padding: 4px 12px; border-radius: 12px; font-weight: 700; border: 1px solid #d3d6d8; }
    
    div[data-testid="stMetricValue"] { font-size: 20px !important; color: #333; }
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
    st.error("❌ Lỗi: Thiếu Secrets SUPABASE_URL hoặc SUPABASE_KEY")
    st.stop()

@st.cache_resource
def init_connection():
    return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_connection()

# --- LOAD AI MODEL ---
@st.cache_resource
def load_ai():
    m_path = MODEL_PATH if os.path.exists(MODEL_PATH) else "lstm_factory_v2.pth"
    s_path = SCALER_PATH if os.path.exists(SCALER_PATH) else "robust_scaler_v2.pkl"
    c_path = CONFIG_PATH if os.path.exists(CONFIG_PATH) else "model_config_v2.pkl"

    if not os.path.exists(m_path): return None, None, None

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
if 'init' not in st.session_state:
    st.session_state.buffer = {(d, c): 0 for d in DEVICES for c in CHANNELS}
    st.session_state.logs = {(d, c): [] for d in DEVICES for c in CHANNELS}
    st.session_state.init = True

# ===============================================================
# 2. XỬ LÝ DỮ LIỆU
# ===============================================================
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
    if df_lane.empty: return 0.0, False, "gray", "NO DATA", "Chờ dữ liệu..."
    
    last = df_lane.iloc[-1]
    # Check Offline
    if (datetime.now() - last['time']).total_seconds() > 180:
        return 0.0, False, "orange", "⚠️ MẤT KẾT NỐI", "Offline > 3 phút"
    
    # Check Dừng
    if last['Speed'] == 0:
        if last.get('Temp', 0) > TEMP_CRASH_THRESHOLD:
            return 9.9, True, "red", "⛔ CRASH", f"Nhiệt cao: {last['Temp']}°C"
        return 0.0, False, "gray", "💤 IDLE", "Máy đang nghỉ"

    # Check AI
    if model:
        loss, is_anom = predict_anomaly(df_lane, model, scaler, config)
        if is_anom:
            stt = "🐢 CHẬM/KẸT" if last['Speed'] < 1.5 else "⚠️ QUÁ TẢI"
            clr = "orange" if last['Speed'] < 1.5 else "red"
            return loss, True, clr, stt, f"AI Loss: {loss:.3f}"
        return loss, False, "green", "✅ ỔN ĐỊNH", "Hoạt động tốt"
    
    return 0.0, False, "gray", "LOADING", "Loading AI..."

# ===============================================================
# 3. UI COMPONENTS
# ===============================================================
def create_gauge(val, title, color):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = val,
        title = {'text': title, 'font': {'size': 15}},
        gauge = {
            'axis': {'range': [None, 5], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'steps': [{'range': [0, 1.5], 'color': '#f0fff4'}, {'range': [1.5, 3.5], 'color': '#dcfce7'}]
        }
    ))
    fig.update_layout(height=140, margin=dict(t=30,b=10,l=20,r=20))
    return fig

def render_lane_card(dev_id, ch, df_lane):
    """Vẽ 1 Làn"""
    now_str = datetime.now().strftime('%H:%M:%S')
    
    if df_lane.empty:
        st.warning(f"Làn {ch}: Chưa có dữ liệu")
        return

    last = df_lane.iloc[-1]
    score, is_danger, color, status_text, log_msg = determine_status(df_lane)

    key = (dev_id, ch)
    if is_danger: st.session_state.buffer[key] += 1
    else: st.session_state.buffer[key] = 0
    final_alert = (st.session_state.buffer[key] >= 2) or ("CRASH" in status_text)

    if final_alert:
        if not st.session_state.logs[key] or st.session_state.logs[key][-1]['msg'] != log_msg:
            st.session_state.logs[key].append({'time': last['time'], 'msg': log_msg})

    css = "status-ok" if color == "green" else ("status-err" if color == "red" else ("status-warn" if color == "orange" else "status-gray"))
    gauge_col = "#10b981" if color == "green" else ("#ef4444" if color == "red" else "#f59e0b")

    # CARD UI
    with st.container(border=True):
        c1, c2 = st.columns([1.5, 1])
        c1.markdown(f"#### 🛣️ Làn {ch}")
        c2.markdown(f'<div class="{css}" style="text-align:center">{status_text}</div>', unsafe_allow_html=True)
        st.divider()
        g_col, m_col = st.columns([1, 1.2])
        with g_col:
            st.plotly_chart(create_gauge(last['Speed'], "Tốc độ", gauge_col), use_container_width=True, key=f"g_{dev_id}_{ch}_{now_str}")
        with m_col:
            st.markdown(f"📦 **SL:** `{int(last['Actual']):,}`")
            st.markdown(f"⏱️ **Run:** `{int(last.get('RunTime',0)/60)}p`")
            st.markdown(f"🌡️ **Temp:** `{last.get('Temp',0):.1f}°C`")
            st.markdown(f"🧠 **Loss:** `{score:.3f}`")
        
        # Sparkline
        chart_data = df_lane.tail(50)
        fig = px.line(chart_data, x='time', y='Speed', height=120)
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), xaxis=dict(visible=False), yaxis=dict(visible=True, range=[0, 5]))
        st.plotly_chart(fig, use_container_width=True, key=f"c_{dev_id}_{ch}_{now_str}")

        with st.expander("📝 Nhật ký", expanded=final_alert):
            if st.session_state.logs[key]:
                l_df = pd.DataFrame(st.session_state.logs[key])
                l_df['time'] = l_df['time'].dt.strftime('%H:%M:%S')
                st.dataframe(l_df.iloc[::-1].head(5), hide_index=True, use_container_width=True)
            else: st.caption("Ổn định.")

# ===============================================================
# 4. MAIN LAYOUT (TAB NGANG)
# ===============================================================
st.title("🏭 STANLEY FACTORY INTELLIGENCE")

# Các TAB nằm ngang
tab1, tab2, tab3 = st.tabs([
    f"🏗️ MÁY 1 ({DEVICES[0][-4:]})", 
    f"🏗️ MÁY 2 ({DEVICES[1][-4:]})", 
    "📊 ANALYTICS"
])

# Lấy dữ liệu 1 lần
df_all = get_recent_data(1000)

# --- TAB 1 ---
with tab1:
    if not df_all.empty:
        col_left, col_right = st.columns(2)
        dev = DEVICES[0]
        # Lấy data và render song song
        df_l1 = df_all[(df_all['DevAddr'] == dev) & (df_all['Channel'] == "01")].sort_values('time')
        df_l2 = df_all[(df_all['DevAddr'] == dev) & (df_all['Channel'] == "02")].sort_values('time')
        
        with col_left: render_lane_card(dev, "01", df_l1)
        with col_right: render_lane_card(dev, "02", df_l2)
    else: st.info("⏳ Đang tải dữ liệu Máy 1...")

# --- TAB 2 ---
with tab2:
    if not df_all.empty:
        col_left, col_right = st.columns(2)
        dev = DEVICES[1]
        
        df_l1 = df_all[(df_all['DevAddr'] == dev) & (df_all['Channel'] == "01")].sort_values('time')
        df_l2 = df_all[(df_all['DevAddr'] == dev) & (df_all['Channel'] == "02")].sort_values('time')
        
        with col_left: render_lane_card(dev, "01", df_l1)
        with col_right: render_lane_card(dev, "02", df_l2)
    else: st.info("⏳ Đang tải dữ liệu Máy 2...")

# --- TAB 3 ---
with tab3:
    st.header("📊 Phân tích & Báo cáo")
    c_sel, _ = st.columns([1, 2])
    with c_sel:
        otp = st.selectbox("Chọn Làn:", [f"{d[-4:]} - Làn {c}" for d in DEVICES for c in CHANNELS])
        days = st.slider("Thời gian:", 1, 30, 7)
        btn = st.button("Tải dữ liệu")
    
    if btn:
        sel_suffix = otp.split(" - ")[0]
        sel_ch = otp.split(" Làn ")[1]
        real_dev = DEVICES[0] if DEVICES[0].endswith(sel_suffix) else DEVICES[1]
        
        start_t = (datetime.utcnow() - timedelta(days=days)).isoformat()
        res = supabase.table("sensor_data").select("time,Speed,Actual,Temp").eq("DevAddr", real_dev).eq("Channel", sel_ch).gte("time", start_t).order("time").execute()
        df_his = pd.DataFrame(res.data)
        
        if not df_his.empty:
            df_his['time'] = pd.to_datetime(df_his['time']).dt.tz_convert('Asia/Bangkok').dt.tz_localize(None)
            k1, k2, k3 = st.columns(3)
            k1.metric("Speed TB", f"{df_his['Speed'].mean():.2f}")
            k2.metric("Tổng SL", f"{df_his['Actual'].max() - df_his['Actual'].min():,}")
            k3.metric("Số bản ghi", f"{len(df_his)}")
            st.plotly_chart(px.line(df_his, x='time', y='Speed', title=f"Biểu đồ Tốc độ: {otp}"), use_container_width=True)
            st.plotly_chart(px.histogram(df_his, x='Speed', title="Phân bố tốc độ"), use_container_width=True)
        else: st.warning("Không có dữ liệu.")

time.sleep(REFRESH_RATE)
st.rerun()