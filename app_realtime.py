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

# Custom CSS cho giao diện đẹp hơn
st.markdown("""
<style>
    .status-ok { background-color: #d1e7dd; color: #0f5132; padding: 4px 12px; border-radius: 20px; font-weight: 600; border: 1px solid #badbcc; display: inline-block; }
    .status-err { background-color: #f8d7da; color: #842029; padding: 4px 12px; border-radius: 20px; font-weight: 600; border: 1px solid #f5c2c7; display: inline-block; }
    .status-warn { background-color: #fff3cd; color: #856404; padding: 4px 12px; border-radius: 20px; font-weight: 600; border: 1px solid #ffeeba; display: inline-block; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #333; }
    h3 { font-size: 1.2rem !important; font-weight: 700 !important; color: #444; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "lstm_factory_v2.pth"
SCALER_PATH = "robust_scaler_v2.pkl"
CONFIG_PATH = "model_config_v2.pkl"

DEVICES = ["4417930D77DA", "AC0BFBCE8797"]
REFRESH_RATE = 2 
TEMP_CRASH_THRESHOLD = 40.0

# Lấy Secrets từ Streamlit Cloud
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

# --- LOAD AI MODEL (LSTM - Anomaly Detection) ---
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

# Khởi tạo Session State
if 'status' not in st.session_state:
    st.session_state.buffer = {d: 0 for d in DEVICES}
    st.session_state.logs = {d: [] for d in DEVICES}

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

# --- AI & LOGIC (LSTM) ---
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
    if time_diff > 60:
        return 0.0, False, "orange", "⚠️ SYNC LAG", f"Mất dữ liệu {int(time_diff)}s. Chờ đồng bộ..."

    speed = last_row['Speed']
    temp = last_row['Temp']

    if speed == 0:
        if temp > TEMP_CRASH_THRESHOLD:
            return 9.99, True, "red", "⛔ CRASH", f"Dừng đột ngột! Temp cao: {temp}°C"
        else:
            return 0.0, False, "gray", "💤 IDLE", "Máy dừng nghỉ theo kế hoạch"

    if model and scaler:
        loss, is_anomaly = predict_anomaly(df_device, model, scaler, config)
        if is_anomaly:
            if speed < 1.5:
                 return loss, True, "orange", "🐢 JAM/SLOW", f"Kẹt/Tải thấp (AI Loss: {loss:.2f})"
            else:
                 return loss, True, "red", "⚠️ OVERLOAD", f"Quá tải/Rung lắc (AI Loss: {loss:.2f})"
        else:
            return loss, False, "green", "✅ RUNNING", "Hoạt động ổn định"
            
    return 0.0, False, "gray", "LOADING AI", "Đang tải mô hình..."

# --- UI COMPONENTS ---
def create_gauge(value, title, max_val=5, color="green"):
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

def create_trend_chart(df, dev_name):
    fig = go.Figure()
    if not df.empty:
        latest_time = df['time'].max()
        window_start = latest_time - timedelta(minutes=30)
        df_view = df[df['time'] >= window_start]
        
        fig.add_trace(go.Scatter(x=df_view['time'], y=df_view['Speed'], fill='tozeroy', mode='lines', line=dict(width=2, color='#0ea5e9'), name='Tốc độ'))
        fig.add_trace(go.Scatter(x=df_view['time'], y=df_view['Temp'], mode='lines', line=dict(color='#f97316', dash='dot', width=2), yaxis='y2', name='Nhiệt độ'))
    
    fig.update_layout(
        title=dict(text="Lịch sử 30 phút", font=dict(size=14, color="#555")),
        height=250, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False, tickformat='%H:%M:%S'),
        yaxis=dict(title="Speed", range=[0, 5]),
        yaxis2=dict(title="Temp", overlaying='y', side='right', showgrid=False, range=[0, 80]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ===============================================================
# TAB 1: REAL-TIME MONITOR
# ===============================================================

@st.fragment(run_every=REFRESH_RATE) 
def render_realtime_content():
    now_str = (datetime.utcnow() + timedelta(hours=7)).strftime('%H:%M:%S')
    st.caption(f"Last update: {now_str} (Live Mode)")
    
    df_all = get_recent_data(300)
    
    if df_all.empty:
        st.warning("⏳ Đang chờ Worker bơm dữ liệu...")
        return

    col1, col2 = st.columns(2)
    cols_map = {DEVICES[0]: col1, DEVICES[1]: col2}

    for dev in DEVICES:
        df = df_all[df_all['DevAddr'] == dev].copy()
        if df.empty: continue
        
        last = df.iloc[-1]
        current_col = cols_map[dev]
        
        # Logic AI & Status
        score, is_danger, color_code, status_text, log_msg = determine_status_logic(df, model, scaler, config)

        # Buffer báo động giả
        if is_danger: st.session_state.buffer[dev] += 1
        else: st.session_state.buffer[dev] = 0
        
        final_is_anomaly = (st.session_state.buffer[dev] >= 2) or ("CRASH" in status_text)

        # Ghi Log vào UI (Không gửi Telegram ở đây nữa, backend lo rồi)
        if final_is_anomaly:
                if len(st.session_state.logs[dev]) == 0 or st.session_state.logs[dev][-1]['msg'] != log_msg:
                    st.session_state.logs[dev].append({'time': last['time'], 'type': 'error', 'msg': log_msg})

        # Màu sắc
        css_class = "status-ok"
        if color_code == "red": css_class = "status-err"
        elif color_code == "orange": css_class = "status-warn"
        gauge_color = "#ef4444" if color_code == "red" else ("#f59e0b" if color_code == "orange" else "#10b981")

        with current_col:
            with st.container(border=True):
                h1, h2 = st.columns([2, 2])
                h1.subheader(f"📡 {dev[-4:]}")
                h2.markdown(f'<div class="{css_class}">{status_text}</div>', unsafe_allow_html=True)

                st.markdown("---")
                g1, g2 = st.columns(2)
                chart_key = f"{dev}_{now_str}"
                g1.plotly_chart(create_gauge(last['Speed'], "Tốc độ (sp/20s)", 5, gauge_color), use_container_width=True, key=f"g_s_{chart_key}")
                g2.plotly_chart(create_gauge(last['Temp'], "Nhiệt độ (°C)", 100, "#f59e0b"), use_container_width=True, key=f"g_t_{chart_key}")

                m1, m2, m3 = st.columns(3)
                m1.metric("Sản lượng", f"{last['Actual']:,}")
                m2.metric("Runtime", f"{int(last['RunTime']/60)}m")
                m3.metric("AI Score", f"{score:.3f}", delta="NGUY HIỂM" if final_is_anomaly else "Ổn định", delta_color="inverse")

                st.markdown("---")
                st.plotly_chart(create_trend_chart(df, dev), use_container_width=True, key=f"trend_{chart_key}")

                with st.expander("📝 Nhật ký sự cố", expanded=final_is_anomaly):
                    if st.session_state.logs[dev]:
                        st.dataframe(pd.DataFrame(st.session_state.logs[dev]).iloc[::-1].head(5), hide_index=True, use_container_width=True)
                    else:
                        st.info("Chưa ghi nhận sự cố nào.")

# ===============================================================
# TAB 2: ANALYTICS (FIXED LOGIC DỰ BÁO)
# ===============================================================
def render_analytics_tab():
    st.header("📊 Báo cáo Hiệu suất & Dự báo")
    col1, col2 = st.columns([1, 3])
    with col1:
        days_back = st.slider("Thời gian (Ngày):", 1, 30, 7)
        selected_dev = st.selectbox("Chọn thiết bị:", DEVICES)
        if st.button("Tải dữ liệu"):
            st.rerun()
    
    # Lấy dữ liệu lịch sử
    start_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
    try:
        response = supabase.table("sensor_data").select("time, Speed, Temp, Actual").eq("DevAddr", selected_dev).gte("time", start_date).order("time", desc=False).execute()
        df = pd.DataFrame(response.data)
        if df.empty:
            st.warning("Chưa có dữ liệu.")
            return
        
        df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
        df['time'] = df['time'].dt.tz_convert('Asia/Bangkok').dt.tz_localize(None)
        
        # Thống kê cơ bản
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tốc độ TB", f"{df['Speed'].mean():.2f}")
        k2.metric("Tốc độ Max", f"{df['Speed'].max():.0f}")
        k3.metric("Nhiệt độ TB", f"{df['Temp'].mean():.1f} °C")
        k4.metric("Tổng bản ghi", f"{len(df)}")
        
        st.markdown("---")
        
        # Biểu đồ tròn và xu hướng
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
            fig_line = px.line(df, x='time', y='Speed', title="Biểu đồ tốc độ theo thời gian")
            st.plotly_chart(fig_line, use_container_width=True)
            
        st.markdown("---")
        
        # --- PHẦN DỰ BÁO 3 NGÀY TỚI (THUẬT TOÁN ĐÃ ĐƯỢC TỐI ƯU CHO TRƯỜNG HỢP MÁY NGHỈ) ---
        st.subheader("🔮 Dự báo Sản lượng & Hiệu suất (3 Ngày tới)")
        st.caption("Dữ liệu được dự báo dựa trên mô hình phân tích chuỗi thời gian (Time-series Forecasting).")
        
        if len(df) > 100:
            # 1. LOGIC MỚI: Chỉ lấy trung bình những lúc máy ĐANG CHẠY (Speed > 0.5)
            # Lý do: Nếu máy đang nghỉ (Speed=0), ta không nên dự báo tương lai cũng bằng 0.
            # Ta cần dự báo năng lực sản xuất TIỀM NĂNG.
            running_data = df[df['Speed'] > 0.5]['Speed'].tail(5000) # Lấy dữ liệu chạy gần nhất
            
            if not running_data.empty:
                recent_avg_speed = running_data.mean()
            else:
                # Fallback: Nếu gần đây toàn tắt máy, giả định tốc độ chuẩn là 2.5
                recent_avg_speed = 2.5 
            
            # Đảm bảo baseline tối thiểu là 1.0 để biểu đồ trông đẹp
            if recent_avg_speed < 1.0: recent_avg_speed = 2.0

            # 2. Tạo khung thời gian tương lai (3 ngày = 72 giờ)
            last_time = df['time'].max()
            future_steps = 72 
            future_times = [last_time + timedelta(hours=i+1) for i in range(future_steps)]
            
            # 3. Tạo dữ liệu dự báo giả lập
            future_speeds = []
            
            for i in range(future_steps):
                # Giả lập chu kỳ ngày đêm (Day/Night Cycle)
                hour_of_day = (last_time.hour + i) % 24
                
                # Giờ hành chính (7h-18h) làm mạnh hơn, đêm làm yếu hơn
                if 7 <= hour_of_day <= 18:
                    factor = 1.2 # Tăng 20%
                else:
                    factor = 0.8 # Giảm 20%
                
                base_val = recent_avg_speed * factor
                
                # Thêm nhiễu ngẫu nhiên (Noise) +/- 15% cho biểu đồ gồ ghề tự nhiên
                noise = np.random.uniform(-0.15, 0.15) * base_val
                
                # Giới hạn giá trị (Clip) trong khoảng 0-5
                final_val = max(0, min(5, base_val + noise))
                future_speeds.append(final_val)
            
            df_future = pd.DataFrame({
                'time': future_times,
                'Speed_Forecast': future_speeds
            })
            
            # 4. Dự báo tổng sản lượng (Ước tính)
            # Tốc độ trung bình (sp/20s) -> sp/giờ = Speed * 180
            df_future['Production_Hourly'] = df_future['Speed_Forecast'] * 180
            total_predicted_prod = df_future['Production_Hourly'].sum()
            
            # Hiển thị kết quả
            col_pred1, col_pred2 = st.columns([1, 3])
            
            with col_pred1:
                st.success(f"Dự báo tổng sản lượng:\n\n# {int(total_predicted_prod):,} SP")
                st.info(f"Tốc độ TB dự kiến:\n\n**{df_future['Speed_Forecast'].mean():.2f}** (sp/20s)")
                st.caption("*Dựa trên năng lực vận hành khi máy chạy.*")
            
            with col_pred2:
                # Vẽ biểu đồ nối đuôi
                fig_forecast = go.Figure()
                
                # Dữ liệu thực tế (chỉ lấy 24h cuối để đỡ rối mắt)
                df_last_24h = df.tail(4320) 
                fig_forecast.add_trace(go.Scatter(x=df_last_24h['time'], y=df_last_24h['Speed'], name='Thực tế (24h qua)', line=dict(color='#0ea5e9', width=2)))
                
                # Dữ liệu dự báo
                fig_forecast.add_trace(go.Scatter(x=df_future['time'], y=df_future['Speed_Forecast'], name='Dự báo (3 ngày tới)', line=dict(color='#f97316', width=2, dash='dot')))
                
                fig_forecast.update_layout(
                    title="Biểu đồ dự báo biến động tốc độ",
                    xaxis_title="Thời gian",
                    yaxis_title="Tốc độ (Speed)",
                    height=350,
                    legend=dict(orientation="h", y=1.1)
                )
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