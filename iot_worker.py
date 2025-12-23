import os
import random
import time
import pandas as pd
from datetime import datetime, timedelta
from supabase import create_client
import openmeteo_requests
import requests_cache
from retry_requests import retry

print("🤖 IOT WORKER: Bắt đầu bơm dữ liệu CHUẨN + SỰ CỐ (4 Lanes)...")

# --- LẤY KEY TỪ MÔI TRƯỜNG ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL:
    print("❌ Lỗi: Thiếu Key Supabase!")
    exit()

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"❌ Lỗi kết nối Supabase: {e}")
    exit()

# CẤU HÌNH 4 LÀN (2 Máy x 2 Kênh)
DEVICES = ["4417930D77DA", "AC0BFBCE8797"]
CHANNELS = ["01", "02"]

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
    # --- CẤU HÌNH ---
    INTERVAL_SECONDS = 20  
    POINTS_PER_RUN = 60    # 20 phút dữ liệu mỗi lần chạy
    
    base_temp, base_hum = get_weather()
    all_payloads = []
    
    start_time_base = datetime.now() - timedelta(seconds=POINTS_PER_RUN * INTERVAL_SECONDS)

    # LOOP QUA TỪNG MÁY VÀ TỪNG KÊNH
    for dev_id in DEVICES:
        for ch in CHANNELS:
            
            # 1. Lấy trạng thái cũ riêng của từng Làn
            curr_actual = 1000000; curr_runtime = 5000000; curr_heldtime = 2000000
            try:
                # Query phải lọc cả DevAddr VÀ Channel
                res = supabase.table("sensor_data")\
                    .select("*")\
                    .eq("DevAddr", dev_id)\
                    .eq("Channel", ch)\
                    .order("time", desc=True)\
                    .limit(1)\
                    .execute()
                    
                if res.data:
                    last = res.data[0]
                    curr_actual = last['Actual']
                    curr_runtime = last['RunTime']
                    curr_heldtime = last['HeldTime']
            except: pass

            # 2. Sinh dữ liệu cho làn này
            for i in range(POINTS_PER_RUN):
                point_time = start_time_base + timedelta(seconds=(i + 1) * INTERVAL_SECONDS)
                
                rand_val = random.random()
                
                # Logic mô phỏng (Giữ nguyên logic của bro)
                if rand_val < 0.05: # CRASH
                    status = 2
                    speed = 0
                    d_runtime = 0.0
                    d_heldtime = float(INTERVAL_SECONDS)
                    temp = base_temp + random.uniform(20.0, 30.0)
                elif rand_val < 0.30: # IDLE
                    status = 1 
                    speed = 0
                    d_runtime = 0.0
                    d_heldtime = float(INTERVAL_SECONDS)
                    temp = base_temp + random.uniform(0.5, 2.0)
                else: # RUNNING
                    status = 1
                    speed = random.choices([0, 1, 2], weights=[0.2, 0.75, 0.05])[0]
                    d_runtime = float(INTERVAL_SECONDS)
                    d_heldtime = 0.0
                    temp = base_temp + random.uniform(5.0, 10.0)
                
                curr_actual += speed
                curr_runtime += d_runtime
                curr_heldtime += d_heldtime
                
                record = {
                    "time": point_time.isoformat(),
                    "DevAddr": dev_id, 
                    "Channel": ch,          # <--- Quan trọng
                    "Actual": curr_actual, 
                    "Status": status,
                    "RunTime": float(curr_runtime), 
                    "HeldTime": float(curr_heldtime),
                    "Speed": float(speed),
                    "d_RunTime": d_runtime,
                    "d_HeldTime": d_heldtime,
                    "Temp": float(f"{temp:.2f}"), 
                    "Humidity": base_hum
                }
                all_payloads.append(record)

    # 3. Gửi Batch lên Supabase
    if all_payloads:
        try:
            # Gửi từng gói 1000 dòng để tránh quá tải
            batch_size = 1000
            for i in range(0, len(all_payloads), batch_size):
                batch = all_payloads[i:i + batch_size]
                supabase.table("sensor_data").insert(batch).execute()
                
            print(f"✅ Đã bơm {len(all_payloads)} dòng dữ liệu cho 4 Làn!")
        except Exception as e:
            print(f"❌ Lỗi insert: {e}")

# --- ENTRY POINT ---
if __name__ == "__main__":
    run_worker_batch()