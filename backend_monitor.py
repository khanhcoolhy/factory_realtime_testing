import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import requests
from supabase import create_client
from datetime import datetime, timedelta

# ===============================================================
# 1. CẤU HÌNH & KHỞI TẠO
# ===============================================================
print("🕵️ MONITOR: Khởi động hệ thống giám sát Backend (Mapping Làn 1-4)...")

MODEL_PATH = "saved_models_v2/lstm_factory_v2.pth"
SCALER_PATH = "saved_models_v2/robust_scaler_v2.pkl"
CONFIG_PATH = "saved_models_v2/model_config_v2.pkl"

# --- MAPPING CẤU HÌNH (SỬA PHẦN NÀY ĐỂ KHỚP APP) ---
# Logic: Ánh xạ từ (Device ID + Channel vật lý) -> Tên Làn hiển thị
LANE_MAPPING = {
    "4417930D77DA": {"01": "Làn 1", "02": "Làn 2"},  # Máy 1
    "AC0BFBCE8797": {"01": "Làn 3", "02": "Làn 4"}   # Máy 2
}

# Lấy danh sách để loop
DEVICES = list(LANE_MAPPING.keys())
CHANNELS = ["01", "02"] # Channel vật lý từ DB

TEMP_CRASH_THRESHOLD = 40.0 

# Lấy Secrets
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Lỗi: Thiếu Key Supabase (Set environment variable)")
    exit()

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"❌ Lỗi kết nối DB: {e}")
    exit()

# ===============================================================
# 2. LOAD AI MODEL (Giống hệt Notebook & App)
# ===============================================================
def load_ai():
    if not os.path.exists(MODEL_PATH):
        print("⚠️ Không tìm thấy file model. Chạy chế độ Rule-based.")
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
        print("✅ Đã load xong AI Model v2")
        return model, scl, cfg
    except Exception as e:
        print(f"❌ Lỗi load AI: {e}")
        return None, None, None

model, scaler, config = load_ai()

# ===============================================================
# 3. HÀM XỬ LÝ
# ===============================================================
def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("   [Log] Chưa cấu hình Telegram.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"❌ Lỗi gửi Telegram: {e}")

def check_system():
    print(f"\n--- Quét lúc {datetime.now().strftime('%H:%M:%S')} ---")
    
    # Lấy dữ liệu mới nhất (đủ cho 4 làn)
    try:
        response = supabase.table("sensor_data").select("*").order("time", desc=True).limit(500).execute()
        df = pd.DataFrame(response.data)
        if df.empty:
            print("⚠️ Không có dữ liệu.")
            return
        
        # Convert Time
        df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
        # Sort để lấy tail chính xác
        df = df.sort_values('time')
    except Exception as e:
        print(f"❌ Lỗi query DB: {e}")
        return

    # Loop qua từng thiết bị và từng kênh
    for dev_id in DEVICES:
        for ch in CHANNELS:
            # Lấy tên hiển thị (Làn 1, 2, 3, 4)
            lane_name = LANE_MAPPING.get(dev_id, {}).get(ch, f"Unknown-{ch}")
            
            # Lọc data cho làn này
            df_lane = df[(df['DevAddr'] == dev_id) & (df['Channel'] == ch)]
            
            if df_lane.empty:
                continue
                
            last_row = df_lane.iloc[-1]
            
            # --- KIỂM TRA LOGIC ---
            
            # 1. Offline Check
            time_diff = (datetime.now(last_row['time'].tzinfo) - last_row['time']).total_seconds()
            if time_diff > 300: # 5 phút
                print(f"⚠️ {lane_name}: Mất kết nối ({int(time_diff)}s)")
                # (Tùy chọn: Gửi cảnh báo offline)
                continue

            # 2. Rule-based Crash Check (Quan trọng)
            if last_row['Speed'] == 0 and last_row['Temp'] > TEMP_CRASH_THRESHOLD:
                msg = (
                    f"🔥 **CẢNH BÁO NGUY HIỂM - {lane_name}**\n"
                    f"---------------\n"
                    f"🌡️ Nhiệt độ: {last_row['Temp']}°C (Quá cao)\n"
                    f"🛑 Trạng thái: Dừng máy đột ngột\n"
                    f"⏰ Thời gian: {last_row['time'].strftime('%H:%M:%S')}"
                )
                print(f"   -> 🔴 {lane_name}: CRASH DETECTED!")
                send_telegram(msg)
                continue # Đã crash thì không check AI nữa

            # 3. AI Anomaly Check
            if model and len(df_lane) > 31: # Cần đủ sequence length
                try:
                    features = config['features_list']
                    data_segment = df_lane[features].tail(31).values
                    
                    # Preprocessing giống Training
                    data_log = np.log1p(data_segment)
                    data_scaled = scaler.transform(data_log)
                    
                    X_input = torch.tensor(data_scaled[:-1], dtype=torch.float32).unsqueeze(0)
                    Y_actual = data_scaled[-1]
                    
                    with torch.no_grad():
                        Y_pred = model(X_input).numpy()[0]
                    
                    target_idx = config.get('target_cols_idx', [0, 1, 2])
                    loss = np.mean(np.abs(Y_pred[target_idx] - Y_actual[target_idx]))
                    
                    # Ngưỡng (Threshold)
                    threshold = config['threshold']
                    
                    if loss > threshold:
                        err_type = "🐢 Kẹt tải / Chậm" if last_row['Speed'] < 1.5 else "⚠️ Rung lắc / Quá tải"
                        msg = (
                            f"🤖 **PHÁT HIỆN BẤT THƯỜNG - {lane_name}**\n"
                            f"---------------\n"
                            f"📉 AI Loss: **{loss:.4f}** (Ngưỡng: {threshold:.3f})\n"
                            f"🔧 Phán đoán: {err_type}\n"
                            f"🏎️ Tốc độ: {last_row['Speed']} m/s\n"
                            f"🌡️ Nhiệt độ: {last_row['Temp']}°C"
                        )
                        print(f"   -> 🟠 {lane_name}: AI ANOMALY (Loss: {loss:.3f})")
                        send_telegram(msg)
                    else:
                        print(f"   -> ✅ {lane_name}: Ổn định (Loss: {loss:.3f})")
                        
                except Exception as e:
                    print(f"   -> ⚠️ Lỗi tính toán AI cho {lane_name}: {e}")

# ===============================================================
# 4. LOOP CHÍNH
# ===============================================================
if __name__ == "__main__":
    while True:
        check_system()
        time.sleep(10) # 10 giây quét 1 lần