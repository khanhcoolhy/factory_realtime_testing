import os
import random
import time
import pandas as pd
from datetime import datetime, timedelta
from supabase import create_client
import openmeteo_requests
import requests_cache
from retry_requests import retry

print("🤖 IOT WORKER: Bắt đầu bơm dữ liệu Siêu Mịn (60 điểm/5 phút)...")

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
    # --- CẤU HÌNH SIÊU MỊN ---
    POINTS_PER_RUN = 60      # 60 điểm dữ liệu
    INTERVAL_SECONDS = 5     # Cách nhau 5 giây
    # Tổng cộng: 60 * 5s = 300s = 5 Phút (Vừa khít lịch chạy GitHub)
    
    base_temp, base_hum = get_weather()
    all_payloads = []
    
    # Bắt đầu từ 5 phút trước cho đến hiện tại
    start_time_base = datetime.now() - timedelta(minutes=5)

    for dev in DEVICES:
        dev_id = dev['id']
        ch = dev['ch']
        
        # Lấy trạng thái cũ để cộng dồn
        curr_actual = 1000000; curr_runtime = 5000000; curr_heldtime = 2000000
        try:
            res = supabase.table("sensor_data").select("*").eq("DevAddr", dev_id).order("time", desc=True).limit(1).execute()
            if res.data:
                last = res.data[0]
                curr_actual = last['Actual']; curr_runtime = last['RunTime']; curr_heldtime = last['HeldTime']
        except: pass

        # Vòng lặp sinh 60 điểm
        for i in range(POINTS_PER_RUN):
            point_time = start_time_base + timedelta(seconds=(i + 1) * INTERVAL_SECONDS)
            
            # Logic Random
            chance = 0.95 if dev_id == "4417930D77DA" else 0.98
            is_anomaly = random.random() > chance
            speed = random.randint(150, 250) if is_anomaly else random.randint(0, 5)
            
            temp = base_temp + random.uniform(-0.5, 0.5)
            
            # Cộng dồn
            curr_actual += speed
            curr_runtime += (20 if speed > 0 else 0)
            curr_heldtime += (20 if speed == 0 else 0)
            
            record = {
                "time": point_time.isoformat(),
                "DevAddr": dev_id, "Channel": ch,
                "Actual": curr_actual, "Status": 1 if speed > 0 else 2,
                "RunTime": curr_runtime, "HeldTime": curr_heldtime,
                "Speed": float(speed),
                "d_RunTime": 20.0 if speed > 0 else 0.0,
                "d_HeldTime": 20.0 if speed == 0 else 0.0,
                "Temp": float(f"{temp:.2f}"), "Humidity": base_hum
            }
            all_payloads.append(record)

    # Gửi 1 cục lên Supabase
    if all_payloads:
        try:
            supabase.table("sensor_data").insert(all_payloads).execute()
            print(f"✅ Đã bơm {len(all_payloads)} điểm dữ liệu (High Resolution).")
        except Exception as e:
            print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    run_worker_batch()