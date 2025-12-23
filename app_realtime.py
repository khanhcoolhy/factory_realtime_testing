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
    .status-warn { background-color: #fff3cd; color: #856404; padding: 4px 12px; border-radius: 20px; font-weight: 600; border: 1px solid #ffeeba; display: inline-block; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #333; }
    h3 { font-size: 1.1rem !important; font-weight: 700 !important; color: #444; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "lstm_factory_v2.pth"
SCALER_PATH = "robust_scaler_v2.pkl"
CONFIG_PATH = "model_config_v2.pkl"

# --- [FIX UI 1] CẤU HÌNH DISPLAY CHO TỪNG MÁY & LÀN ---
DEVICES_CONFIG = [
    {"id": "4417930D77DA", "name": "MÁY HÀN 01", "channels": ["01", "02"]},
    {"id": "AC0BFBCE8797", "name": "MÁY DẬP 02", "channels": ["01", "02"]}
]

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

# --- LOAD AI MODEL (LSTM) ---
@st.cache_resource
def load_ai():
    if not os.path.exists(MODEL_PATH): return None, None, None
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
        return None, None, None

model, scaler, config = load_ai()

# --- [FIX STATE] State Management cho từng Làn ---
if 'status' not in st.session_state:
    st.session_state.buffer = {} # Key sẽ là "DevID_Channel"
    st.session_state.logs = {}   # Key sẽ là "DevID_Channel"

# --- HÀM HỖ TRỢ ---
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

# --- AI LOGIC ---
def predict_anomaly(df_device, model, scaler, config):
    SEQ_LEN = 30
    if len(df_device) < SEQ_LEN + 1: return 0.0, False
    
    features = config['features_list']
    try:
        data_segment = df_device[features].tail(SEQ_LEN + 1).values
    except KeyError:
        return 0.0, False
        
    data_log = np.log1p(data_segment)
    data_scaled = scaler.transform(data_log)
    
    X_input = data_scaled[:-1]
    Y_actual = data_scaled[-1]
    
    X_tensor = torch.tensor(X_input, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        Y_pred = model(X_tensor).numpy()[0]
        
    target_idx = config.get('target_cols_idx', [0, 1, 2])
    loss = np.mean(np.abs(Y_pred[target_idx] - Y_actual[target_idx]))
    
    threshold = config['threshold']
    is_anomaly = loss > threshold
    return loss, is_anomaly

def determine_status_logic(df_device, model, scaler, config):
    if df_device.empty or len(df_device) < 2:
        return 0.0, False, "gray", "NO DATA", "Chưa có dữ liệu"

    last_row = df_device.iloc[-1]
    prev_row = df_device.iloc[-2]
    
    time_diff = (last_row['time'] - prev_row['time']).total_seconds()
    # Tăng time check lên chút vì dữ liệu gửi mỗi 20s
    if time_diff > 120:
        return 0.0, False, "orange", "⚠️ SYNC LAG", f"Mất kết nối {int(time_diff)}s"

    speed = last_row['Speed']
    temp = last_row['Temp']

    if speed == 0:
        if temp > TEMP_CRASH_THRESHOLD:
            return 9.99, True, "red", "⛔ CRASH", f"Dừng đột ngột! Temp: {temp}°C"
        else:
            return 0.0, False, "gray", "💤 IDLE", "Máy đang nghỉ"

    if model and scaler:
        loss, is_anomaly = predict_anomaly(df_device, model, scaler, config)
        if is_anomaly:
            if speed < 1.5:
                 return loss, True, "orange", "🐢 SLOW/JAM", f"Tải thấp/Kẹt (Loss: {loss:.2f})"
            else:
                 return loss, True, "red", "⚠️ OVERLOAD", f"Quá tải (Loss: {loss:.2f})"
        else:
            return loss, False, "green", "✅ RUNNING", "Hoạt động ổn định"
            
    return 0.0, False, "gray", "LOADING AI", "Đang tải mô hình..."

# --- UI COMPONENTS ---
def create_gauge(value, title, max_val=5, color="green"):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = value,
        title = {'text': title, 'font': {'size': 14, 'color': '#555'}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white", 'borderwidth': 1, 'bordercolor': "#ddd",
            'steps': [{'range': [0, max_val*0.3], 'color': '#f0fff4'}, {'range': [max_val*0.3, max_val*0.7], 'color': '#dcfce7'}, {'range': [max_val*0.7, max_val], 'color': '#bbf7d0'}],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': max_val * 0.9}
        }
    ))
    fig.update_layout(height=160, margin=dict(t=30,b=10,l=25,r=25))
    return fig

def create_trend_chart(df, title_suffix):
    fig = go.Figure()
    if not df.empty:
        latest_time = df['time'].max()
        window_start = latest_time - timedelta(minutes=20)
        df_view = df[df['time'] >= window_start]
        
        fig.add_trace(go.Scatter(x=df_view['time'], y=df_view['Speed'], fill='tozeroy', mode='lines', line=dict(width=2, color='#0ea5e9'), name='Tốc độ'))
        fig.add_trace(go.Scatter(x=df_view['time'], y=df_view['Temp'], mode='lines', line=dict(color='#f97316', dash='dot', width=1.5), yaxis='y2', name='Nhiệt độ'))
    
    fig.update_layout(
        title=dict(text=f"Biến động {title_suffix}", font=dict(size=12, color="#555")),
        height=200, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showgrid=False, tickformat='%H:%M:%S'),
        yaxis=dict(title="Speed", range=[0, 5], showticklabels=False),
        yaxis2=dict(overlaying='y', side='right', showgrid=False, range=[0, 80], showticklabels=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ===============================================================
# TAB 1: REAL-TIME MONITOR (ĐÃ SỬA UI)
# ===============================================================
@st.fragment(run_every=REFRESH_RATE) 
def render_realtime_content():
    now_str = (datetime.utcnow() + timedelta(hours=7)).strftime('%H:%M:%S')
    st.caption(f"Last update: {now_str} (Live Mode)")
    
    # Lấy dữ liệu 1 lần cho tối ưu
    df_all = get_recent_data(500)
    
    if df_all.empty:
        st.warning("⏳ Đang chờ Worker bơm dữ liệu...")
        return

    # --- [FIX UI 2] Loop qua từng Device, rồi loop qua từng Channel ---
    for dev_conf in DEVICES_CONFIG:
        d_id = dev_conf['id']
        d_name = dev_conf['name']
        channels = dev_conf['channels']
        
        st.subheader(f"🏭 {d_name} ({d_id[-4:]})")
        
        # Tạo số cột tương ứng với số kênh (Làn)
        cols = st.columns(len(channels))
        
        for idx, ch in enumerate(channels):
            with cols[idx]:
                # Tạo khóa duy nhất cho lane này
                lane_key = f"{d_id}_{ch}"
                
                # Init Session State cho lane nếu chưa có
                if lane_key not in st.session_state.buffer:
                    st.session_state.buffer[lane_key] = 0
                    st.session_state.logs[lane_key] = []

                # --- [QUAN TRỌNG] Filter dữ liệu CHỈ CỦA CHANNEL NÀY ---
                # Đây là bước sửa lỗi biểu đồ zig-zag
                df_lane = df_all[
                    (df_all['DevAddr'] == d_id) & 
                    (df_all['Channel'] == ch)
                ].copy()
                
                # Logic Xử lý
                score, is_danger, color_code, status_text, log_msg = determine_status_logic(df_lane, model, scaler, config)

                # Debounce Buffer
                if is_danger: st.session_state.buffer[lane_key] += 1
                else: st.session_state.buffer[lane_key] = 0
                
                final_is_anomaly = (st.session_state.buffer[lane_key] >= 2) or ("CRASH" in status_text)

                # Ghi Log
                if final_is_anomaly:
                    last_log = st.session_state.logs[lane_key][-1] if st.session_state.logs[lane_key] else None
                    if not last_log or last_log['msg'] != log_msg:
                        st.session_state.logs[lane_key].append({'time': datetime.now().strftime('%H:%M:%S'), 'msg': log_msg})

                # Style CSS
                css_class = "status-ok"
                if color_code == "red": css_class = "status-err"
                elif color_code == "orange": css_class = "status-warn"
                gauge_color = "#ef4444" if color_code == "red" else ("#f59e0b" if color_code == "orange" else "#10b981")

                # --- VẼ GIAO DIỆN CHO 1 LANE ---
                with st.container(border=True):
                    # Header Lane
                    c1, c2 = st.columns([2, 2])
                    c1.markdown(f"**Làn (Lane) {ch}**")
                    c2.markdown(f'<div class="{css_class}">{status_text}</div>', unsafe_allow_html=True)
                    
                    if not df_lane.empty:
                        last = df_lane.iloc[-1]
                        
                        # Gauge & Metric
                        g1, g2 = st.columns(2)
                        chart_id = f"{lane_key}_{now_str}"
                        g1.plotly_chart(create_gauge(last['Speed'], "Tốc độ", 5, gauge_color), use_container_width=True, key=f"g1_{chart_id}")
                        
                        with g2:
                            st.metric("Sản lượng", f"{last['Actual']:,}")
                            st.metric("Nhiệt độ", f"{last['Temp']}°C")

                        # Biểu đồ Trend nhỏ
                        st.plotly_chart(create_trend_chart(df_lane, f"Làn {ch}"), use_container_width=True, key=f"tr_{chart_id}")
                        
                        # Log sự cố
                        if final_is_anomaly:
                            st.error(f"⚠️ {log_msg}")

                    else:
                        st.info("Chưa có dữ liệu làn này")

        st.markdown("---") # Ngăn cách giữa các máy

# ===============================================================
# TAB 2: ANALYTICS (ĐÃ SỬA CHỌN LANE)
# ===============================================================
def render_analytics_tab():
    st.header("📊 Báo cáo Hiệu suất & Dự báo")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        # Chọn Máy (Hiển thị tên cho đẹp)
        dev_options = {d['name']: d['id'] for d in DEVICES_CONFIG}
        selected_name = st.selectbox("Chọn thiết bị:", list(dev_options.keys()))
        selected_dev_id = dev_options[selected_name]
    
    with col2:
        # Chọn Làn (Dynamic theo máy)
        # Tìm config của máy đang chọn
        curr_conf = next(item for item in DEVICES_CONFIG if item["id"] == selected_dev_id)
        selected_channel = st.selectbox("Chọn Làn (Channel):", curr_conf['channels'])

    with col3:
        days_back = st.slider("Ngày:", 1, 30, 7)
        if st.button("Tải dữ liệu"): st.rerun()
    
    # Lấy dữ liệu lịch sử CÓ LỌC CHANNEL
    start_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
    try:
        response = supabase.table("sensor_data")\
            .select("time, Speed, Temp, Actual, Channel")\
            .eq("DevAddr", selected_dev_id)\
            .eq("Channel", selected_channel)\
            .gte("time", start_date)\
            .order("time", desc=False)\
            .execute()
            
        df = pd.DataFrame(response.data)
        if df.empty:
            st.warning("Chưa có dữ liệu cho Làn này.")
            return
        
        df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
        df['time'] = df['time'].dt.tz_convert('Asia/Bangkok').dt.tz_localize(None)
        
        # --- PHẦN DƯỚI GIỮ NGUYÊN LOGIC CŨ NHƯNG DATA ĐÃ SẠCH ---
        # Thống kê cơ bản
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tốc độ TB", f"{df['Speed'].mean():.2f}")
        k2.metric("Tốc độ Max", f"{df['Speed'].max():.0f}")
        k3.metric("Nhiệt độ TB", f"{df['Temp'].mean():.1f} °C")
        k4.metric("Tổng bản ghi", f"{len(df)}")
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("⏱️ Tỷ lệ Vận hành")
            conditions = [(df['Speed'] == 0), (df['Speed'] > 0)]
            choices = ['Dừng (Idle)', 'Hoạt động (Running)']
            df['State'] = np.select(conditions, choices, default='Không rõ')
            state_counts = df['State'].value_counts().reset_index()
            state_counts.columns = ['State', 'Count']
            fig_pie = px.pie(state_counts, values='Count', names='State', hole=0.4, color='State', color_discrete_map={'Dừng (Idle)': '#9e9e9e', 'Hoạt động (Running)': '#2ecc71'})
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.subheader("📈 Xu hướng Quá khứ")
            fig_line = px.line(df, x='time', y='Speed', title=f"Tốc độ Làn {selected_channel}")
            st.plotly_chart(fig_line, use_container_width=True)
            
        st.markdown("---")
        
        # --- DỰ BÁO 3 NGÀY (GIỮ NGUYÊN LOGIC) ---
        st.subheader(f"🔮 Dự báo Làn {selected_channel} (3 Ngày tới)")
        
        if len(df) > 100:
            running_data = df[df['Speed'] > 0.5]['Speed'].tail(5000)
            recent_avg_speed = running_data.mean() if not running_data.empty else 2.5
            std_dev = df['Speed'].tail(1000).std()
            if pd.isna(std_dev) or std_dev == 0: std_dev = recent_avg_speed * 0.1

            last_time = df['time'].max()
            future_steps = 72 
            future_times = [last_time + timedelta(hours=i+1) for i in range(future_steps)]
            
            future_speeds = []
            for i in range(future_steps):
                hour_of_day = (last_time.hour + i) % 24
                if 7 <= hour_of_day <= 18: factor = 1.1 
                else: factor = 0.9 
                base_val = recent_avg_speed * factor
                noise = np.random.uniform(-0.5, 0.5) * std_dev
                final_val = max(0, base_val + noise)
                future_speeds.append(final_val)
            
            df_future = pd.DataFrame({'time': future_times, 'Speed_Forecast': future_speeds})
            
            col_pred1, col_pred2 = st.columns([1, 3])
            with col_pred1:
                st.success(f"Dự kiến sản lượng:\n\n# {int(df_future['Speed_Forecast'].sum() * 180):,} SP")
                st.info(f"Tốc độ TB:\n\n**{df_future['Speed_Forecast'].mean():.2f}**")
            
            with col_pred2:
                fig_forecast = go.Figure()
                df_last_24h = df.tail(1000) 
                fig_forecast.add_trace(go.Scatter(x=df_last_24h['time'], y=df_last_24h['Speed'], name='Thực tế', line=dict(color='#0ea5e9', width=1)))
                fig_forecast.add_trace(go.Bar(
                    x=df_future['time'], y=df_future['Speed_Forecast'], name='Dự báo', 
                    marker=dict(color='#f97316', opacity=0.7),
                    hovertemplate='Thời gian: %{x}<br>Tốc độ: %{y:.2f}<extra></extra>'
                ))
                fig_forecast.update_layout(title="Dự báo Tốc độ", height=400, barmode='overlay', legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_forecast, use_container_width=True)
        else:
            st.info("⚠️ Cần ít nhất 100 điểm dữ liệu để chạy mô hình dự báo.")

    except Exception as e:
        st.error(f"Lỗi hiển thị báo cáo: {e}")

# ===============================================================
# MAIN
# ===============================================================
st.title("🏭 STANLEY FACTORY INTELLIGENCE")
st.markdown("---")

tab1, tab2 = st.tabs(["🚀 REAL-TIME MONITOR", "📈 ANALYTICS"])

with tab1:
    render_realtime_content()

with tab2:
    render_analytics_tab()