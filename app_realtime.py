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
import traceback # Thêm thư viện để bắt lỗi chi tiết
from datetime import datetime, timedelta
from supabase import create_client

# ===============================================================
# 1. CẤU HÌNH HỆ THỐNG & KẾT NỐI
# ===============================================================
st.set_page_config(page_title="Hệ thống Giám sát Nhà máy", layout="wide", page_icon="🏭")

# --- CSS ---
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #f0f2f6; padding-top: 20px; }
    .stRadio > div { background-color: white; padding: 10px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .stRadio label { font-weight: bold; font-size: 16px; padding: 10px 5px; }
    div[data-testid="stMetricValue"] { font-size: 22px !important; }
    .status-badge { padding: 5px 10px; border-radius: 6px; font-weight: bold; text-align: center; color: white; }
</style>
""", unsafe_allow_html=True)

# --- CONFIG THIẾT BỊ ---
MACHINE_CONFIG = {
    "MÁY 1": {
        "id": "4417930D77DA",
        "name": "MÁY CẮT 01",
        "lanes": [
            {"code": "L1", "name": "Làn 1", "db_channel": "01"},
            {"code": "L2", "name": "Làn 2", "db_channel": "02"}
        ]
    },
    "MÁY 2": {
        "id": "AC0BFBCE8797",
        "name": "MÁY CẮT 02",
        "lanes": [
            {"code": "L3", "name": "Làn 3", "db_channel": "01"},
            {"code": "L4", "name": "Làn 4", "db_channel": "02"}
        ]
    }
}

MODEL_PATH = "saved_models_v2/lstm_factory_v2.pth"
SCALER_PATH = "saved_models_v2/robust_scaler_v2.pkl"
CONFIG_PATH = "saved_models_v2/model_config_v2.pkl"
REFRESH_RATE = 2 

# --- SUPABASE (Thêm Try/Except để tránh crash ngay từ đầu) ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"❌ Lỗi kết nối Supabase hoặc thiếu Secrets: {e}")
    st.stop()

# --- LOAD AI MODEL ---
@st.cache_resource
def load_ai():
    if not os.path.exists(CONFIG_PATH): 
        # Không tìm thấy file thì trả về None nhưng không crash app
        return None, None, None
    try:
        cfg = joblib.load(CONFIG_PATH)
        scl = joblib.load(SCALER_PATH)
        
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

if 'logs' not in st.session_state:
    st.session_state.logs = []

# ===============================================================
# 2. XỬ LÝ DỮ LIỆU (FIXED)
# ===============================================================
def get_realtime_data(limit=500): 
    try:
        response = supabase.table("sensor_data").select("*").order("time", desc=True).limit(limit).execute()
        df = pd.DataFrame(response.data)
        if not df.empty:
            # Sửa lỗi ValueError do format datetime
            df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True, errors='coerce')
            df = df.dropna(subset=['time']) # Bỏ dòng lỗi thời gian
            df['time'] = df['time'].dt.tz_convert('Asia/Bangkok').dt.tz_localize(None)
            df = df.sort_values('time')
        return df
    except Exception as e:
        st.error(f"Lỗi lấy dữ liệu: {e}")
        return pd.DataFrame()

def predict_anomaly(df_lane, model, scaler, config):
    SEQ_LEN = 30
    # Kiểm tra an toàn
    if len(df_lane) < SEQ_LEN + 1: return 0.0, False
    if config is None or model is None: return 0.0, False
    
    try:
        features = config.get('features_list', [])
        # Kiểm tra xem đủ cột không
        if not all(col in df_lane.columns for col in features):
            return 0.0, False
            
        data_segment = df_lane[features].tail(SEQ_LEN + 1).values
        
        # FIX: Kiểm tra NaN trước khi đưa vào scaler (Nguyên nhân chính gây ValueError)
        if np.isnan(data_segment).any():
            return 0.0, False

        data_log = np.log1p(data_segment)
        data_scaled = scaler.transform(data_log)
        
        X_input = torch.tensor(data_scaled[:-1], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad(): 
            Y_pred = model(X_input).numpy()[0]
            
        Y_actual = data_scaled[-1]
        
        target_idx = config.get('target_cols_idx', [0, 1, 2])
        loss = np.mean(np.abs(Y_pred[target_idx] - Y_actual[target_idx]))
        
        return loss, loss > config.get('threshold', 0.1)
    except Exception:
        # Nuốt lỗi AI để app không sập
        return 0.0, False

def analyze_lane_status(df_lane):
    if df_lane.empty: return "NODATA", "black", "Chờ dữ liệu", 0.0
    
    try:
        last = df_lane.iloc[-1]
        now = datetime.now()
        
        # Check Offline
        if (now - last['time']).total_seconds() > 300: # 5 phút
            return "OFFLINE", "gray", f"Mất kết nối {last['time'].strftime('%H:%M')}", 0.0

        # Check Crash
        if last['Speed'] == 0 and last.get('Temp', 0) > 40:
            return "CRASH", "red", f"⚠️ QUÁ NHIỆT {last['Temp']}°C", 9.9

        # Check Idle
        if last['Speed'] < 0.1:
            return "IDLE", "#6c757d", "💤 Máy đang nghỉ", 0.0

        loss, is_anom = predict_anomaly(df_lane, model, scaler, config)
        
        if is_anom:
            if last['Speed'] < 1.5: return "SLOW", "orange", "🐢 Chạy chậm", loss
            return "OVERLOAD", "red", "🔥 Quá tải", loss
            
        return "OK", "green", "✅ Ổn định", loss
    except Exception as e:
        return "ERR", "gray", "Lỗi xử lý", 0.0

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
    st_code, color, msg, ai_loss = analyze_lane_status(df_lane)
    
    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        c1.markdown(f"### 🛣️ {lane_cfg['name']}")
        c2.markdown(f'<div style="background-color:{color};" class="status-badge">{msg}</div>', unsafe_allow_html=True)
        st.divider()
        
        if not df_lane.empty:
            last = df_lane.iloc[-1]
            kc1, kc2 = st.columns([1, 1])
            with kc1:
                st.plotly_chart(draw_gauge(last['Speed'], "Tốc độ (m/s)", color), use_container_width=True)
            with kc2:
                st.metric("📦 Sản lượng", f"{int(last['Actual']):,}")
                st.metric("🌡️ Nhiệt độ", f"{last.get('Temp', 0):.1f}°C")
                st.metric("🧠 AI Loss", f"{ai_loss:.4f}")

            # Sparkline
            chart_df = df_lane.tail(50)
            fig = px.area(chart_df, x='time', y='Speed', height=100)
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
            fig.update_traces(line_color=color, fillcolor=color, fill_opacity=0.1)
            st.plotly_chart(fig, use_container_width=True)

# ===============================================================
# 4. MAIN LAYOUT (CÓ BẮT LỖI TOÀN CỤC)
# ===============================================================
try:
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2620/2620952.png", width=70)
        st.title("STANLEY IoT")
        st.caption(f"Update: {datetime.now().strftime('%H:%M:%S')}")
        st.markdown("---")
        
        # Bỏ tham số captions để tránh lỗi phiên bản cũ
        selected_tab = st.radio(
            "KHU VỰC GIÁM SÁT:", 
            ["🏗️ MÁY 1 (Làn 1-2)", "🏗️ MÁY 2 (Làn 3-4)", "📊 ANALYTICS"]
        )
        
        st.markdown("---")
        if st.session_state.logs:
            st.warning("🔔 Log gần nhất:")
            st.caption(st.session_state.logs[-1])

    df_all = get_realtime_data(1000)

    if "MÁY 1" in selected_tab:
        cfg = MACHINE_CONFIG["MÁY 1"]
        st.header(f"📡 {cfg['name']} - Realtime Monitor")
        col_left, col_right = st.columns(2)
        
        df_dev = df_all[df_all['DevAddr'] == cfg['id']] if not df_all.empty else pd.DataFrame()
        
        with col_left:
            lane_info = cfg['lanes'][0]
            df_l = df_dev[df_dev['Channel'] == lane_info['db_channel']].sort_values('time') if not df_dev.empty else pd.DataFrame()
            render_lane_view(lane_info, df_l)
            
        with col_right:
            lane_info = cfg['lanes'][1]
            df_l = df_dev[df_dev['Channel'] == lane_info['db_channel']].sort_values('time') if not df_dev.empty else pd.DataFrame()
            render_lane_view(lane_info, df_l)

    elif "MÁY 2" in selected_tab:
        cfg = MACHINE_CONFIG["MÁY 2"]
        st.header(f"📡 {cfg['name']} - Realtime Monitor")
        col_left, col_right = st.columns(2)
        
        df_dev = df_all[df_all['DevAddr'] == cfg['id']] if not df_all.empty else pd.DataFrame()
        
        with col_left:
            lane_info = cfg['lanes'][0]
            df_l = df_dev[df_dev['Channel'] == lane_info['db_channel']].sort_values('time') if not df_dev.empty else pd.DataFrame()
            render_lane_view(lane_info, df_l)
            
        with col_right:
            lane_info = cfg['lanes'][1]
            df_l = df_dev[df_dev['Channel'] == lane_info['db_channel']].sort_values('time') if not df_dev.empty else pd.DataFrame()
            render_lane_view(lane_info, df_l)

    else:
        st.header("📊 Phân Tích & Báo Cáo")
        st.info("Chức năng đang phát triển...")

    # Refresh
    time.sleep(REFRESH_RATE)
    st.rerun()

except Exception as e:
    st.error("❌ ĐÃ CÓ LỖI XẢY RA!")
    # In chi tiết lỗi ra màn hình để debug
    st.code(traceback.format_exc())
    st.stop()