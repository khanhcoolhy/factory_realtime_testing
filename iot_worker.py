import os
import random
import time
import pandas as pd
from datetime import datetime, timedelta
from supabase import create_client
import openmeteo_requests
import requests_cache
from retry_requests import retry

print("🤖 IOT WORKER: Bắt đầu bơm dữ liệu CHUẨN + SỰ CỐ (Simulation)...")

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
    # --- CẤU HÌNH ---
    INTERVAL_SECONDS = 20  
    POINTS_PER_RUN = 60    # Sinh 20 phút dữ liệu mỗi lần chạy
    
    base_temp, base_hum = get_weather()
    all_payloads = []
    
    start_time_base = datetime.now() - timedelta(seconds=POINTS_PER_RUN * INTERVAL_SECONDS)

    for dev in DEVICES:
        dev_id = dev['id']
        ch = dev['ch']
        
        # 1. Lấy trạng thái cũ
        curr_actual = 1000000; curr_runtime = 5000000; curr_heldtime = 2000000
        try:
            res = supabase.table("sensor_data").select("*").eq("DevAddr", dev_id).order("time", desc=True).limit(1).execute()
            if res.data:
                last = res.data[0]
                curr_actual = last['Actual']
                curr_runtime = last['RunTime']
                curr_heldtime = last['HeldTime']
        except: pass

        # 2. Vòng lặp sinh dữ liệu
        for i in range(POINTS_PER_RUN):
            point_time = start_time_base + timedelta(seconds=(i + 1) * INTERVAL_SECONDS)
            
            # --- LOGIC MÔ PHỎNG 3 TRẠNG THÁI ---
            rand_val = random.random()
            
            # Kịch bản phân phối:
            # 70% Chạy bình thường
            # 25% Nghỉ (Idle)
            # 5%  Sự cố (Crash) -> Để test App
            
            if rand_val < 0.05: 
                # === TRƯỜNG HỢP 1: CRASH (SỰ CỐ) ===
                # Logic: Speed = 0 NHƯNG Nhiệt độ RẤT CAO
                status = 2 # Error
                speed = 0
                d_runtime = 0.0
                d_heldtime = float(INTERVAL_SECONDS) # Tính vào thời gian dừng
                # Nhiệt độ cao hơn môi trường rất nhiều (> 40 độ để trigger App)
                temp = base_temp + random.uniform(20.0, 30.0) 

            elif rand_val < 0.30:
                # === TRƯỜNG HỢP 2: IDLE (NGHỈ) ===
                status = 1 # Hoặc trạng thái chờ
                speed = 0
                d_runtime = 0.0
                d_heldtime = float(INTERVAL_SECONDS)
                # Nhiệt độ mát (nguội dần về nhiệt độ môi trường)
                temp = base_temp + random.uniform(0.5, 2.0)
                
            else:
                # === TRƯỜNG HỢP 3: RUNNING (CHẠY) ===
                status = 1
                speed = random.choices([0, 1, 2], weights=[0.2, 0.75, 0.05])[0]
                d_runtime = float(INTERVAL_SECONDS)
                d_heldtime = 0.0
                # Nhiệt độ ấm (do máy chạy)
                temp = base_temp + random.uniform(5.0, 10.0)
            
            # Cập nhật cộng dồn
            curr_actual += speed
            curr_runtime += d_runtime
            curr_heldtime += d_heldtime
            
            record = {
                "time": point_time.isoformat(),
                "DevAddr": dev_id, 
                "Channel": ch,
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

    # 3. Gửi lên Supabase
    if all_payloads:
        try:
            supabase.table("sensor_data").insert(all_payloads).execute()
            print(f"✅ Đã bơm {len(all_payloads)} điểm dữ liệu (bao gồm cả CRASH test)!")
        except Exception as e:
            print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    # Chạy vòng lặp để bơm liên tục mỗi 20s (giả lập realtime)
    # Hoặc chạy 1 lần rồi thôi tùy bạn. Ở đây mình để loop để bạn test App cho sướng.
    while True:
        run_worker_batch()
        print("😴 Nghỉ 20s trước khi bơm tiếp batch mới (để App kịp hiển thị)...")
        time.sleep(20)