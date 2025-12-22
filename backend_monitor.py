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
print("🕵️ MONITOR: Khởi động hệ thống giám sát Backend (AI + GSheet)...")

# --- FILE MODEL ---
MODEL_PATH = "lstm_factory_v2.pth"
SCALER_PATH = "robust_scaler_v2.pkl"
CONFIG_PATH = "model_config_v2.pkl"

# --- THIẾT BỊ ---
DEVICES = ["4417930D77DA", "AC0BFBCE8797"]
TEMP_CRASH_THRESHOLD = 40.0 

# --- GOOGLE SHEET URL (Của bạn) ---
GSHEET_URL = "https://script.google.com/macros/s/AKfycbx-NbALoc4_iisA-rQO5Z1uFzfh1HYo6B2y4e_FlFqyCV0y_bQRiILYa2LjMbQhZ9uI/exec"

# --- SECRETS (GITHUB ACTIONS) ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Lỗi: Thiếu Key SUPABASE!")
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
        print("✅ Đã load xong Model AI & Scaler.")
        return model, scl, cfg
    except Exception as e:
        print(f"❌ Lỗi load AI Model: {e}")
        return None, None, None

model, scaler, config = load_ai()

# ===============================================================
# 3. CÁC HÀM CẢNH BÁO (ALERTS)
# ===============================================================

def send_telegram(msg):
    """Gửi tin nhắn cảnh báo qua Telegram"""
    if not TELEGRAM_TOKEN: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=5)
        print("📨 Đã gửi cảnh báo Telegram.")
    except:
        print("⚠️ Không thể gửi Telegram.")

def save_to_google_sheet(dev_id, error_type, message, score):
    """Lưu nhật ký sự cố vào Google Sheet qua Apps Script"""
    try:
        payload = {
            "dev_id": dev_id,
            "type": error_type,
            "message": message,
            "score": float(score)
        }
        # Gửi POST request tới Google Script
        resp = requests.post(GSHEET_URL, json=payload, timeout=10)
        if resp.status_code == 200:
            print(f"🗒️ Đã lưu nhật ký vào Google Sheet thành công.")
        else:
            print(f"⚠️ GSheet trả lỗi: {resp.status_code}")
    except Exception as e:
        print(f"❌ Lỗi khi ghi Google Sheet: {e}")

# ===============================================================
# 4. LOGIC KIỂM TRA CHÍNH
# ===============================================================
def check_device_status(dev_id):
    print(f"\n🔍 Đang kiểm tra thiết bị: {dev_id}...")
    
    try:
        # 1. Lấy dữ liệu mới nhất từ Supabase
        response = supabase.table("sensor_data")\
            .select("*")\
            .eq("DevAddr", dev_id)\
            .order("time", desc=True)\
            .limit(40)\
            .execute()
            
        df = pd.DataFrame(response.data)
        if df.empty: return
        
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        last_row = df.iloc[-1]
        
        # 2. Check dữ liệu quá cũ (trên 25 phút)
        now_utc = datetime.utcnow()
        time_diff = (now_utc - last_row['time'].replace(tzinfo=None)).total_seconds()
        if time_diff > 1500:
            print(f"   -> 💤 Dữ liệu quá cũ, bỏ qua.")
            return

        # 3. KIỂM TRA CRASH (Rule-based)
        if last_row['Speed'] == 0:
            if last_row['Temp'] > TEMP_CRASH_THRESHOLD:
                # PHÁT HIỆN LỖI
                msg_content = f"Dừng đột ngột! Temp: {last_row['Temp']}°C"
                full_msg = (
                    f"🚨 **CẢNH BÁO SỰ CỐ (CRASH)**\n"
                    f"🤖 Thiết bị: `{dev_id}`\n"
                    f"⚠️ Lỗi: {msg_content}\n"
                    f"🕒 Lúc: {last_row['time'].strftime('%H:%M:%S')}"
                )
                print("   -> 🔴 PHÁT HIỆN CRASH!")
                send_telegram(full_msg)
                # Ghi vào Google Sheet
                save_to_google_sheet(dev_id, "CRASH", msg_content, 0.0)
            else:
                print("   -> 💤 Máy nghỉ (Idle).")
            return

        # 4. KIỂM TRA AI (Anomaly Detection)
        SEQ_LEN = 30
        if len(df) < SEQ_LEN + 1 or model is None: return

        features = config['features_list']
        data_segment = df[features].tail(SEQ_LEN + 1).values
        data_log = np.log1p(data_segment)
        data_scaled = scaler.transform(data_log)
        
        X_input = torch.tensor(data_scaled[:-1], dtype=torch.float32).unsqueeze(0)
        Y_actual = data_scaled[-1]
        
        with torch.no_grad():
            Y_pred = model(X_input).numpy()[0]
        
        target_idx = config.get('target_cols_idx', [0, 1, 2])
        loss = np.mean(np.abs(Y_pred[target_idx] - Y_actual[target_idx]))
        
        if loss > config['threshold']:
            # PHÁT HIỆN LỖI
            err_type = "Kẹt tải/Chậm" if last_row['Speed'] < 1.5 else "Quá tải/Rung lắc"
            msg_content = f"AI phát hiện bất thường: {err_type}"
            full_msg = (
                f"⚠️ **PHÁT HIỆN BẤT THƯỜNG (AI)**\n"
                f"🤖 Thiết bị: `{dev_id}`\n"
                f"📉 AI Score: {loss:.3f}\n"
                f"🔧 Loại: {err_type}\n"
                f"🕒 Lúc: {last_row['time'].strftime('%H:%M:%S')}"
            )
            print(f"   -> 🟠 BẤT THƯỜNG AI (Loss: {loss:.3f})")
            send_telegram(full_msg)
            # Ghi vào Google Sheet
            save_to_google_sheet(dev_id, "AI_ANOMALY", msg_content, loss)
        else:
            print(f"   -> ✅ Bình thường (Loss: {loss:.3f})")

    except Exception as e:
        print(f"❌ Lỗi: {e}")

# ===============================================================
# 5. MAIN
# ===============================================================
if __name__ == "__main__":
    for dev in DEVICES:
        check_device_status(dev)
    print("\n🏁 Hoàn tất phiên giám sát.")