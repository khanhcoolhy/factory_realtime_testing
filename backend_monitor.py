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
print("🕵️ MONITOR: Khởi động hệ thống giám sát Backend (4 LANES)...")

# Cập nhật đường dẫn theo cấu trúc thư mục của Notebook
MODEL_PATH = "saved_models_v2/lstm_factory_v2.pth"
SCALER_PATH = "saved_models_v2/robust_scaler_v2.pkl"
CONFIG_PATH = "saved_models_v2/model_config_v2.pkl"

# Danh sách Cặp (Máy, Làn) cần giám sát
DEVICES = ["4417930D77DA", "AC0BFBCE8797"]
CHANNELS = ["01", "02"] # 4 Làn tổng cộng

TEMP_CRASH_THRESHOLD = 40.0 

# Lấy Secrets
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Lỗi: Thiếu Key SUPABASE!")
    exit(1)

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"❌ Lỗi kết nối Supabase: {e}")
    exit(1)

# ===============================================================
# 2. LOAD AI MODEL
# ===============================================================
def load_ai():
    # Fallback đường dẫn nếu chạy local hoặc server
    m_path = MODEL_PATH if os.path.exists(MODEL_PATH) else "lstm_factory_v2.pth"
    s_path = SCALER_PATH if os.path.exists(SCALER_PATH) else "robust_scaler_v2.pkl"
    c_path = CONFIG_PATH if os.path.exists(CONFIG_PATH) else "model_config_v2.pkl"

    if not os.path.exists(m_path): 
        print(f"❌ Không tìm thấy model tại {m_path}")
        return None, None, None
    
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
        
        print("✅ Đã load xong Model AI (4 Lanes Ready).")
        return model, scl, cfg
    except Exception as e:
        print(f"❌ Lỗi khi load AI Model: {e}")
        return None, None, None

model, scaler, config = load_ai()

# ===============================================================
# 3. HÀM GỬI TELEGRAM
# ===============================================================
def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=5)
    except: pass

# ===============================================================
# 4. LOGIC KIỂM TRA (PER LANE)
# ===============================================================
def check_lane_status(dev_id, channel):
    print(f"\n🔍 Đang kiểm tra: {dev_id} - Kênh {channel}...")
    
    try:
        # 1. Query Supabase: Lọc cả DevAddr VÀ Channel
        response = supabase.table("sensor_data")\
            .select("*")\
            .eq("DevAddr", dev_id)\
            .eq("Channel", channel)\
            .order("time", desc=True)\
            .limit(40)\
            .execute()
            
        df = pd.DataFrame(response.data)
        
        if df.empty: 
            print("   -> ⚠️ Không có dữ liệu.")
            return
        
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time') # Quan trọng: Sort đúng thứ tự cho LSTM
        last_row = df.iloc[-1]
        
        # 2. Staleness Check
        now_utc = datetime.utcnow()
        last_time_utc = last_row['time'].replace(tzinfo=None)
        if (now_utc - last_time_utc).total_seconds() > 1500:
            print(f"   -> 💤 Dữ liệu cũ. Bỏ qua.")
            return

        # 3. Rule Based Check (Crash)
        if last_row['Speed'] == 0:
            if last_row.get('Temp', 0) > TEMP_CRASH_THRESHOLD:
                msg = (
                    f"🚨 **CẢNH BÁO CRASH - LÀN {channel}**\n"
                    f"---------------\n"
                    f"🤖 Máy: `{dev_id}`\n"
                    f"🔥 Nhiệt độ: **{last_row['Temp']}°C**\n"
                    f"🛑 Tốc độ: 0\n"
                    f"🕒 Lúc: {last_row['time'].strftime('%H:%M:%S')}\n"
                    f"⚠️ *Dừng đột ngột, nhiệt độ cao!*"
                )
                print("   -> 🔴 PHÁT HIỆN CRASH!")
                send_telegram(msg)
            else:
                print("   -> 💤 Máy nghỉ (Idle).")
            return 

        # 4. AI Anomaly Detection
        SEQ_LEN = 30
        if len(df) < SEQ_LEN + 1:
            print("   -> ⚠️ Chưa đủ dữ liệu để chạy AI.")
            return
            
        if model is None: return

        features = config['features_list']
        data_segment = df[features].tail(SEQ_LEN + 1).values
        
        # Log Transform & Scale (Phải khớp Notebook)
        data_log = np.log1p(data_segment)
        data_scaled = scaler.transform(data_log)
        
        X_input = torch.tensor(data_scaled[:-1], dtype=torch.float32).unsqueeze(0)
        Y_actual = data_scaled[-1]
        
        with torch.no_grad():
            Y_pred = model(X_input).numpy()[0]
        
        target_idx = config.get('target_cols_idx', [0, 1, 2])
        loss = np.mean(np.abs(Y_pred[target_idx] - Y_actual[target_idx]))
        
        if loss > config['threshold']:
            err_type = "🐢 Kẹt tải / Chậm" if last_row['Speed'] < 1.5 else "⚠️ Quá tải / Rung"
            msg = (
                f"⚠️ **BẤT THƯỜNG AI - LÀN {channel}**\n"
                f"---------------\n"
                f"🤖 Máy: `{dev_id[-4:]}`\n"
                f"📉 Loss: **{loss:.3f}** (Limit: {config['threshold']:.2f})\n"
                f"🔧 Lỗi: {err_type}\n"
                f"🏎️ Speed: {last_row['Speed']}\n"
            )
            print(f"   -> 🟠 BẤT THƯỜNG AI (Loss: {loss:.3f})")
            send_telegram(msg)
        else:
            print(f"   -> ✅ Ổn định (Loss: {loss:.3f})")

    except Exception as e:
        print(f"❌ Lỗi xử lý {dev_id}-{channel}: {e}")

# ===============================================================
# 5. MAIN LOOP
# ===============================================================
if __name__ == "__main__":
    # Duyệt qua từng Máy và từng Làn
    for dev in DEVICES:
        for ch in CHANNELS:
            check_lane_status(dev, ch)
    
    print("\n🏁 Hoàn tất kiểm tra 4 Làn.")