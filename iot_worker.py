import os
import random
import time
import pandas as pd
from datetime import datetime, timedelta
from supabase import create_client
import openmeteo_requests
import requests_cache
from retry_requests import retry

print("🤖 IOT WORKER: Bắt đầu bơm dữ liệu CHĂM CHỈ (High Performance)...")

# --- LẤY KEY TỪ MÔI TRƯỜNG ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL:
    print("❌ Lỗi: Thiếu Key Supabase!")
    exit()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

DEVICES = [
    {"id": "4417930D77DA", "ch": "01"},
    {"id": "AC0BFBCE8797", "ch": "02"}
]

# API Thời tiết
def get_weather():
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

def run_worker_batch():
    # Cấu hình: 60 điểm/5 phút
    POINTS_PER_RUN = 60
    INTERVAL_SECONDS = 5
    
    base_temp, base_hum = get_weather()
    all_payloads = []
    
    start_time_base = datetime.now() - timedelta(minutes=5)

    for dev in DEVICES:
        dev_id = dev['id']
        ch = dev['ch']
        
        # Lấy trạng thái cũ
        curr_actual = 1000000; curr_runtime = 5000000; curr_heldtime = 2000000
        try:
            res = supabase.table("sensor_data").select("*").eq("DevAddr", dev_id).order("time", desc=True).limit(1).execute()
            if res.data:
                last = res.data[0]
                curr_actual = last['Actual']; curr_runtime = last['RunTime']; curr_heldtime = last['HeldTime']
        except: pass

        for i in range(POINTS_PER_RUN):
            point_time = start_time_base + timedelta(seconds=(i + 1) * INTERVAL_SECONDS)
            
            # --- SỬA LOGIC TẠI ĐÂY ---
            # Máy chạy ổn định 95% thời gian (Speed cao)
            # Chỉ dừng/lỗi 5% thời gian (Speed thấp)
            is_running = random.random() < 0.95 
            
            if is_running:
                # Máy chạy: Tốc độ dao động từ 180 đến 240 (Nhìn cho mạnh)
                speed = random.randint(180, 240)
            else:
                # Máy dừng: Tốc độ về 0 hoặc rất thấp
                speed = random.randint(0, 5)
            
            # Nhiệt độ tăng theo tốc độ
            temp = base_temp + (speed / 300 * 15) + random.uniform(-0.5, 0.5)
            
            # Cộng dồn
            curr_actual += int(speed / 12) # Giả sử 12 speed = 1 sản phẩm
            curr_runtime += (INTERVAL_SECONDS if speed > 0 else 0)
            curr_heldtime += (INTERVAL_SECONDS if speed == 0 else 0)
            
            record = {
                "time": point_time.isoformat(),
                "DevAddr": dev_id, "Channel": ch,
                "Actual": curr_actual, "Status": 1 if speed > 10 else 2,
                "RunTime": curr_runtime, "HeldTime": curr_heldtime,
                "Speed": float(speed),
                "d_RunTime": float(INTERVAL_SECONDS) if speed > 0 else 0.0,
                "d_HeldTime": float(INTERVAL_SECONDS) if speed == 0 else 0.0,
                "Temp": float(f"{temp:.2f}"), "Humidity": base_hum
            }
            all_payloads.append(record)

    if all_payloads:
        try:
            supabase.table("sensor_data").insert(all_payloads).execute()
            print(f"✅ Đã bơm {len(all_payloads)} điểm dữ liệu CHĂM CHỈ!")
        except Exception as e:
            print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    run_worker_batch()