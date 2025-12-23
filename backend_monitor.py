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
print("🕵️ MONITOR: Khởi động hệ thống giám sát Backend (Multi-Channel Support)...")

# Các file Model (Phải có sẵn trong repo GitHub)
MODEL_PATH = "lstm_factory_v2.pth"
SCALER_PATH = "robust_scaler_v2.pkl"
CONFIG_PATH = "model_config_v2.pkl"

# --- [FIX LOGIC 1]: Cấu hình thiết bị kèm theo danh sách Channel (Làn) ---
# Logic Notebook: Mỗi máy có 2 làn độc lập -> Phải xử lý riêng từng làn.
DEVICES_CONFIG = [
    {"id": "4417930D77DA", "channels": ["01", "02"]},
    {"id": "AC0BFBCE8797", "channels": ["01", "02"]}
]

# Ngưỡng nhiệt độ để xác định máy chết (Crash) khi Speed = 0
TEMP_CRASH_THRESHOLD = 40.0 

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
        if resp.status_code == 200:
            print("📨 Đã gửi cảnh báo Telegram thành công.")
        else:
            print(f"⚠️ Gửi Telegram thất bại: {resp.text}")
    except Exception as e:
        print(f"❌ Lỗi kết nối Telegram: {e}")

# ===============================================================
# 4. LOGIC KIỂM TRA (CORE) - ĐÃ SỬA LOGIC LANE
# ===============================================================
def check_lane_status(dev_id, channel_id):
    # --- [FIX LOGIC 2]: Hàm này chỉ check cụ thể 1 Làn của 1 Máy ---
    print(f"\n🔍 Đang kiểm tra: {dev_id} - Làn {channel_id}...")
    
    # 1. Lấy dữ liệu từ Supabase
    try:
        # --- [FIX LOGIC 3]: Thêm .eq("Channel", channel_id) để lọc đúng làn ---
        response = supabase.table("sensor_data")\
            .select("*")\
            .eq("DevAddr", dev_id)\
            .eq("Channel", channel_id)\
            .order("time", desc=True)\
            .limit(40)\
            .execute()
            
        df = pd.DataFrame(response.data)
        
        if df.empty: 
            print("   -> ⚠️ Không có dữ liệu trong DB cho làn này.")
            return
        
        # Sắp xếp lại theo thời gian tăng dần (Cũ -> Mới)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        
        # Lấy dòng mới nhất để kiểm tra trạng thái hiện tại
        last_row = df.iloc[-1]
        
        # 2. Kiểm tra tính mới của dữ liệu (Staleness Check)
        now_utc = datetime.utcnow()
        last_time_utc = last_row['time'].replace(tzinfo=None)
        time_diff = (now_utc - last_time_utc).total_seconds()
        
        if time_diff > 1500: # 1500s = 25 phút
            print(f"   -> 💤 Dữ liệu quá cũ ({int(time_diff/60)} phút trước). Bỏ qua.")
            return

        # 3. LOGIC PHÁT HIỆN SỰ CỐ (CRASH) - Rule Based
        if last_row['Speed'] == 0:
            if last_row['Temp'] > TEMP_CRASH_THRESHOLD:
                # --- [FIX LOGIC 4]: Báo rõ Làn nào bị lỗi ---
                msg = (
                    f"🚨 **CẢNH BÁO SỰ CỐ (CRASH)**\n"
                    f"---------------\n"
                    f"🤖 Thiết bị: `{dev_id}`\n"
                    f"🛤️ Làn (Channel): `{channel_id}`\n"
                    f"🌡️ Nhiệt độ: **{last_row['Temp']}°C** (Quá nóng!)\n"
                    f"🛑 Tốc độ: 0\n"
                    f"🕒 Lúc: {last_row['time'].strftime('%H:%M:%S')}\n"
                    f"---------------\n"
                    f"⚠️ *Máy dừng đột ngột, vui lòng kiểm tra ngay!*"
                )
                print("   -> 🔴 PHÁT HIỆN CRASH!")
                send_telegram(msg)
            else:
                print("   -> 💤 Máy đang nghỉ (Idle) - Nhiệt độ thấp.")
            return # Nếu Speed = 0 thì không chạy AI nữa

        # 4. LOGIC AI (Anomaly Detection) - Khi Speed > 0
        SEQ_LEN = 30
        
        # Kiểm tra đủ dữ liệu để chạy AI không
        if len(df) < SEQ_LEN + 1:
            print(f"   -> ⚠️ Không đủ dữ liệu liên tục (Cần {SEQ_LEN+1}, có {len(df)}).")
            return
            
        if model is None:
            print("   -> ⚠️ Model chưa load được, bỏ qua bước AI.")
            return

        # Chuẩn bị dữ liệu cho Model
        features = config['features_list']
        try:
            # Lấy đúng đoạn dữ liệu cuối cùng
            data_segment = df[features].tail(SEQ_LEN + 1).values
        except KeyError as e:
             print(f"   -> ❌ Thiếu cột dữ liệu: {e}")
             return

        # Transform (Log -> Scale)
        data_log = np.log1p(data_segment)
        data_scaled = scaler.transform(data_log)
        
        X_input = torch.tensor(data_scaled[:-1], dtype=torch.float32).unsqueeze(0)
        Y_actual = data_scaled[-1]
        
        # Dự báo
        with torch.no_grad():
            Y_pred = model(X_input).numpy()[0]
        
        # Tính sai số (Loss)
        target_idx = config.get('target_cols_idx', [0, 1, 2])
        loss = np.mean(np.abs(Y_pred[target_idx] - Y_actual[target_idx]))
        
        # So sánh với ngưỡng
        if loss > config['threshold']:
            if last_row['Speed'] < 1.5:
                err_type = "🐢 Kẹt tải / Tốc độ chậm"
            else:
                err_type = "⚠️ Quá tải / Rung lắc"
                
            msg = (
                f"⚠️ **PHÁT HIỆN BẤT THƯỜNG (AI)**\n"
                f"---------------\n"
                f"🤖 Thiết bị: `{dev_id}`\n"
                f"🛤️ Làn (Channel): `{channel_id}`\n"
                f"📉 AI Score: **{loss:.3f}** (Ngưỡng: {config['threshold']:.2f})\n"
                f"🔧 Loại lỗi: {err_type}\n"
                f"🏎️ Tốc độ: {last_row['Speed']}\n"
                f"🕒 Lúc: {last_row['time'].strftime('%H:%M:%S')}\n"
            )
            print(f"   -> 🟠 PHÁT HIỆN BẤT THƯỜNG AI (Loss: {loss:.3f})")
            send_telegram(msg)
        else:
            print(f"   -> ✅ Hoạt động bình thường (Loss: {loss:.3f})")

    except Exception as e:
        print(f"❌ Lỗi không mong muốn với {dev_id}-{channel_id}: {e}")

# ===============================================================
# 5. MAIN LOOP
# ===============================================================
if __name__ == "__main__":
    # --- [FIX LOGIC 5]: Lặp lồng nhau Device -> Channel ---
    for device_conf in DEVICES_CONFIG:
        d_id = device_conf["id"]
        channels = device_conf["channels"]
        
        for ch in channels:
            check_lane_status(d_id, ch)
    
    print("\n🏁 Kết thúc phiên giám sát.")