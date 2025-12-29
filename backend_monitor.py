import os
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
print("🕵️ MONITOR: Khởi động hệ thống giám sát Backend (AI Logic Update)...")

# Các file Model (Phải có sẵn trong repo GitHub/Folder)
MODEL_PATH = "lstm_factory_v2.pth"
SCALER_PATH = "robust_scaler_v2.pkl"
CONFIG_PATH = "model_config_v2.pkl"

# Danh sách thiết bị và các làn (Channel) cần giám sát
DEVICES_CONFIG = [
    {"id": "4417930D77DA", "channels": ["01", "02"]},
    {"id": "AC0BFBCE8797", "channels": ["01", "02"]}
]

# Lấy Secrets từ biến môi trường
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Kiểm tra biến môi trường
if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Lỗi: Thiếu Key SUPABASE_URL hoặc SUPABASE_KEY!")
    exit(1)

# Kết nối Supabase
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"❌ Lỗi kết nối Supabase: {e}")
    exit(1)

# ===============================================================
# 2. LOAD AI MODEL
# ===============================================================
def load_ai():
    if not os.path.exists(MODEL_PATH): 
        print(f"❌ Không tìm thấy file model: {MODEL_PATH}")
        return None, None, None
    
    try:
        # Load Config & Scaler
        cfg = joblib.load(CONFIG_PATH)
        scl = joblib.load(SCALER_PATH)
        
        # Định nghĩa lại kiến trúc mạng LSTM (phải khớp lúc train)
        class LSTMModel(nn.Module):
            def __init__(self, n_features, hidden_dim=128, num_layers=3, dropout=0.2):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(n_features, hidden_dim, num_layers, batch_first=True, dropout=dropout)
                self.fc = nn.Linear(hidden_dim, n_features)
            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return out

        # Load Weights
        model = LSTMModel(n_features=cfg['n_features'], hidden_dim=cfg['hidden_dim'])
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        
        print("✅ Đã load xong Model AI & Scaler.")
        return model, scl, cfg
    except Exception as e:
        print(f"❌ Lỗi khi load AI Model: {e}")
        return None, None, None

# Load model ngay khi script chạy
model, scaler, config = load_ai()

# ===============================================================
# 3. HÀM GỬI TELEGRAM
# ===============================================================
def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: 
        print("⚠️ Không có Token Telegram, bỏ qua gửi tin nhắn.")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID, 
        "text": msg, 
        "parse_mode": "Markdown"
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=5)
        if resp.status_code != 200:
            print(f"⚠️ Gửi Telegram thất bại: {resp.text}")
    except Exception as e:
        print(f"❌ Lỗi kết nối Telegram: {e}")

# ===============================================================
# 4. LOGIC KIỂM TRA (CORE LOGIC ĐÃ ĐƯỢC LÀM MỚI)
# ===============================================================
def check_lane_status(dev_id, channel_id):
    print(f"\n🔍 Đang kiểm tra: {dev_id} - Làn {channel_id}...")
    
    try:
        # 1. Lấy dữ liệu mới nhất từ Supabase cho đúng Làn (Channel)
        response = supabase.table("sensor_data")\
            .select("*")\
            .eq("DevAddr", dev_id)\
            .eq("Channel", channel_id)\
            .order("time", desc=True)\
            .limit(40)\
            .execute()
            
        df = pd.DataFrame(response.data)
        
        if df.empty: 
            print("   -> ⚠️ Không có dữ liệu trong DB.")
            return
        
        # Sắp xếp lại theo thời gian tăng dần (để chạy Sequence Time Series)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        last_row = df.iloc[-1]
        
        # 2. Kiểm tra Mất kết nối (Offline Check)
        now_utc = datetime.utcnow()
        last_time_utc = last_row['time'].replace(tzinfo=None)
        time_diff = (now_utc - last_time_utc).total_seconds()
        
        if time_diff > 1500: # 25 phút
            print(f"   -> ❌ Thiết bị mất kết nối ({int(time_diff/60)} phút trước).")
            return

        # 3. Kiểm tra Máy Dừng (Idle Check)
        # Nếu Speed = 0, ta coi như máy đang nghỉ, không cần chạy AI
        if last_row['Speed'] == 0:
            print("   -> 💤 Máy đang nghỉ (Speed=0). Bỏ qua AI.")
            return

        # 4. LOGIC AI (Anomaly Detection)
        # -----------------------------------------------------------
        SEQ_LEN = 30
        
        # Kiểm tra đủ độ dài dữ liệu
        if len(df) < SEQ_LEN + 1:
            print(f"   -> ⚠️ Không đủ dữ liệu liên tục (Cần {SEQ_LEN+1}, có {len(df)}).")
            return
            
        if model is None:
            return # Model chưa load được

        # Chuẩn bị input (Vẫn lấy cả Temp/Humidity để khớp với input model 5 chiều)
        features = config['features_list']
        try:
            data_segment = df[features].tail(SEQ_LEN + 1).values
        except KeyError as e:
             print(f"   -> ❌ Thiếu cột dữ liệu: {e}")
             return

        # Transform (Log -> Scale)
        data_log = np.log1p(data_segment)
        data_scaled = scaler.transform(data_log)
        
        # Tách Input (30 dòng đầu) và Output thực tế (dòng 31)
        X_input = torch.tensor(data_scaled[:-1], dtype=torch.float32).unsqueeze(0)
        Y_actual = data_scaled[-1]
        
        # AI Dự báo
        with torch.no_grad():
            Y_pred = model(X_input).numpy()[0]
        
        # Tính sai số (Loss) trên các cột quan trọng (Speed, RunTime, HeldTime)
        target_idx = config.get('target_cols_idx', [0, 1, 2])
        loss = np.mean(np.abs(Y_pred[target_idx] - Y_actual[target_idx]))
        
        # 5. PHÂN LOẠI LỖI (Dựa trên Hành vi Speed)
        # -----------------------------------------------------------
        if loss > config['threshold']:
            # Tính tốc độ trung bình của chuỗi dữ liệu vừa lấy
            avg_speed_segment = df['Speed'].tail(SEQ_LEN).mean()
            
            # Logic phân loại:
            # - Nếu Speed hiện tại thấp hơn 50% Speed trung bình -> Kẹt tải
            # - Ngược lại (Speed vẫn cao nhưng Loss cao) -> Chạy không đều/Rung lắc
            if last_row['Speed'] < (avg_speed_segment * 0.5):
                err_type = "🐢 Kẹt tải / Tốc độ sụt giảm"
                emoji = "🐢"
            else:
                err_type = "⚠️ Rung lắc / Hoạt động bất ổn"
                emoji = "📉"
                
            msg = (
                f"{emoji} **CẢNH BÁO VẬN HÀNH (AI)**\n"
                f"---------------\n"
                f"🤖 Thiết bị: `{dev_id}`\n"
                f"🛤️ Làn (Channel): `{channel_id}`\n"
                f"🔥 Sai số AI (Loss): **{loss:.3f}** (Ngưỡng: {config['threshold']:.2f})\n"
                f"🔧 Chẩn đoán: {err_type}\n"
                f"🏎️ Tốc độ: {last_row['Speed']} (TB: {avg_speed_segment:.1f})\n"
                f"🕒 Lúc: {last_row['time'].strftime('%H:%M:%S')}\n"
            )
            print(f"   -> 🟠 BẤT THƯỜNG: {err_type} (Loss: {loss:.3f})")
            send_telegram(msg)
        else:
            print(f"   -> ✅ Hoạt động bình thường (Loss: {loss:.3f})")

    except Exception as e:
        print(f"❌ Lỗi Runtime tại {dev_id}-{channel_id}: {e}")

# ===============================================================
# 5. MAIN LOOP
# ===============================================================
if __name__ == "__main__":
    # Lặp qua từng thiết bị và từng kênh của thiết bị đó
    for device_conf in DEVICES_CONFIG:
        d_id = device_conf["id"]
        channels = device_conf["channels"]
        
        for ch in channels:
            check_lane_status(d_id, ch)
    
    print("\n🏁 Kết thúc phiên giám sát.")