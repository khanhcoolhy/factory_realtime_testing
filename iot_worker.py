import os
import random
import time
import pandas as pd
from datetime import datetime
from supabase import create_client
import openmeteo_requests
import requests_cache
from retry_requests import retry

print("🤖 IOT WORKER: Bắt đầu chạy theo lịch...")

# --- LẤY KEY TỪ MÔI TRƯỜNG (GITHUB SECRETS) ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    # Fallback cho chạy local test (nếu cần)
    print("⚠️ Không tìm thấy biến môi trường. Kiểm tra lại config.")
    exit()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# DANH SÁCH MÁY
DEVICES = [
    {"id": "4417930D77DA", "ch": "01"},
    {"id": "AC0BFBCE8797", "ch": "02"}
]

# API Thời tiết
def get_realtime_weather():
    try:
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": 21.02, "longitude": 105.83, "current": ["temperature_2m", "relative_humidity_2m"]}
        res = openmeteo.weather_api(url, params=params)[0]
        curr = res.Current()
        return curr.Variables(0).Value(), curr.Variables(1).Value()
    except: return 25.0, 70.0

def run_worker_once():
    now = datetime.now()
    temp, hum = get_realtime_weather()
    
    for dev in DEVICES:
        dev_id = dev['id']
        ch = dev['ch']
        
        # 1. Lấy trạng thái cũ nhất từ Cloud để cộng dồn
        try:
            res = supabase.table("sensor_data").select("*").eq("DevAddr", dev_id).order("time", desc=True).limit(1).execute()
            
            if res.data:
                last = res.data[0]
                curr_actual = last['Actual']
                curr_runtime = last['RunTime']
                curr_heldtime = last['HeldTime']
            else:
                # Khởi tạo nếu chưa có data
                curr_actual = 100000
                curr_runtime = 500000
                curr_heldtime = 200000
                
            # 2. Sinh dữ liệu giả lập (Logic cũ)
            chance = 0.95 if dev_id == "4417930D77DA" else 0.98
            is_anomaly = random.random() > chance
            speed = random.randint(100, 200) if is_anomaly else random.randint(0, 5)
            
            # Cập nhật cộng dồn
            curr_actual += speed
            curr_runtime += (20 if speed > 0 else 0)
            curr_heldtime += (20 if speed == 0 else 0)
            status = 1 if speed > 0 else 2
            
            # 3. Ghi lên Supabase
            data_payload = {
                "time": now.isoformat(),
                "DevAddr": dev_id, "Channel": ch,
                "Actual": curr_actual, "Status": status,
                "RunTime": curr_runtime, "HeldTime": curr_heldtime,
                "Speed": float(speed),
                "d_RunTime": 20.0 if speed > 0 else 0.0,
                "d_HeldTime": 20.0 if speed == 0 else 0.0,
                "Temp": temp, "Humidity": hum
            }
            
            supabase.table("sensor_data").insert(data_payload).execute()
            print(f"✅ {dev_id} | Uploaded Speed: {speed}")
            
        except Exception as e:
            print(f"❌ Lỗi {dev_id}: {e}")

if __name__ == "__main__":
    run_worker_once()
    print("😴 Xong việc. Worker đi ngủ.")