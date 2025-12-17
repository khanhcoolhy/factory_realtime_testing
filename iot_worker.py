import os
import random
import time
import pandas as pd
from datetime import datetime, timedelta
from supabase import create_client
import openmeteo_requests
import requests_cache
from retry_requests import retry

print("🤖 IOT WORKER: Bắt đầu bơm dữ liệu hàng loạt (Batch)...")

# --- LẤY KEY TỪ MÔI TRƯỜNG ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("⚠️ Chạy Local? Đang tìm key trong .env hoặc hardcode...")
    # Nếu chạy local để test thì bro điền key vào đây, còn trên GitHub thì kệ nó
    # SUPABASE_URL = "https://..."
    # SUPABASE_KEY = "..."

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
    # Cấu hình Batch: Sinh ra 10 điểm dữ liệu (cho 5 phút, mỗi 30s một điểm)
    POINTS_PER_RUN = 10
    INTERVAL_SECONDS = 30
    
    base_temp, base_hum = get_weather()
    all_payloads = [] # Chứa tất cả dữ liệu để bắn 1 lần
    
    start_time_base = datetime.now() - timedelta(minutes=5) # Bắt đầu từ 5 phút trước

    for dev in DEVICES:
        dev_id = dev['id']
        ch = dev['ch']
        
        # 1. Lấy trạng thái CŨ NHẤT hiện tại từ Cloud để cộng dồn tiếp
        curr_actual = 1000000
        curr_runtime = 5000000
        curr_heldtime = 2000000
        
        try:
            res = supabase.table("sensor_data").select("*").eq("DevAddr", dev_id).order("time", desc=True).limit(1).execute()
            if res.data:
                last = res.data[0]
                curr_actual = last['Actual']
                curr_runtime = last['RunTime']
                curr_heldtime = last['HeldTime']
        except: pass

        # 2. Vòng lặp sinh 10 điểm liên tiếp
        for i in range(POINTS_PER_RUN):
            # Tính thời gian cho điểm dữ liệu này (tăng dần 30s)
            point_time = start_time_base + timedelta(seconds=(i + 1) * INTERVAL_SECONDS)
            
            # Logic sinh số liệu ngẫu nhiên (giữ nguyên logic cũ)
            chance = 0.95 if dev_id == "4417930D77DA" else 0.98
            is_anomaly = random.random() > chance
            speed = random.randint(150, 250) if is_anomaly else random.randint(0, 5)
            
            # Biến động nhẹ nhiệt độ cho thật
            temp = base_temp + random.uniform(-0.5, 0.5)
            
            # Cộng dồn
            curr_actual += speed
            curr_runtime += (20 if speed > 0 else 0)
            curr_heldtime += (20 if speed == 0 else 0)
            status = 1 if speed > 0 else 2
            
            # Đóng gói
            record = {
                "time": point_time.isoformat(),
                "DevAddr": dev_id, "Channel": ch,
                "Actual": curr_actual, "Status": status,
                "RunTime": curr_runtime, "HeldTime": curr_heldtime,
                "Speed": float(speed),
                "d_RunTime": 20.0 if speed > 0 else 0.0,
                "d_HeldTime": 20.0 if speed == 0 else 0.0,
                "Temp": float(f"{temp:.2f}"), "Humidity": base_hum
            }
            all_payloads.append(record)

    # 3. Gửi tất cả lên mây 1 lần (Bulk Insert)
    if all_payloads:
        try:
            # Supabase Insert nhận vào một list -> Rất nhanh
            supabase.table("sensor_data").insert(all_payloads).execute()
            print(f"✅ Đã bơm thành công {len(all_payloads)} dòng dữ liệu (Batch 5 phút).")
        except Exception as e:
            print(f"❌ Lỗi Upload: {e}")

if __name__ == "__main__":
    run_worker_batch()