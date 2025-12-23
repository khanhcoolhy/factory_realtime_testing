import os
import random
import time
import pandas as pd
from datetime import datetime, timedelta
from supabase import create_client
import openmeteo_requests
import requests_cache
from retry_requests import retry

print("🤖 IOT WORKER: Bắt đầu bơm dữ liệu (2 Device x 2 Channel = 4 Luồng)...")

# --- LẤY KEY TỪ MÔI TRƯỜNG ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL:
    print("❌ Lỗi: Thiếu Key Supabase!")
    exit()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- [FIX LOGIC 1]: Cấu hình 1 máy có nhiều Làn (Channel) ---
DEVICES = [
    {"id": "4417930D77DA", "channels": ["01", "02"]},
    {"id": "AC0BFBCE8797", "channels": ["01", "02"]}
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
    
    # Lùi thời gian lại để bơm dữ liệu nối tiếp nhau
    start_time_base = datetime.now() - timedelta(seconds=POINTS_PER_RUN * INTERVAL_SECONDS)

    # --- [FIX LOGIC 2]: Vòng lặp lồng nhau Device -> Channel ---
    for dev in DEVICES:
        dev_id = dev['id']
        channels = dev['channels']
        
        for ch in channels:
            # 1. Lấy trạng thái cũ CỦA RIÊNG CHANNEL ĐÓ
            curr_actual = 1000000; curr_runtime = 5000000; curr_heldtime = 2000000
            try:
                # [FIX LOGIC 3]: Thêm .eq("Channel", ch) để lấy đúng dòng cuối của làn này
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

            # 2. Vòng lặp sinh dữ liệu cho Channel này
            for i in range(POINTS_PER_RUN):
                point_time = start_time_base + timedelta(seconds=(i + 1) * INTERVAL_SECONDS)
                
                # --- LOGIC MÔ PHỎNG 3 TRẠNG THÁI (Độc lập cho từng làn) ---
                rand_val = random.random()
                
                # Kịch bản phân phối:
                # 70% Chạy bình thường
                # 25% Nghỉ (Idle)
                # 5%  Sự cố (Crash) -> Để test hệ thống cảnh báo
                
                if rand_val < 0.05: 
                    # === TRƯỜNG HỢP 1: CRASH (SỰ CỐ) ===
                    status = 2 # Error
                    speed = 0
                    d_runtime = 0.0
                    d_heldtime = float(INTERVAL_SECONDS)
                    temp = base_temp + random.uniform(20.0, 30.0) # Nóng

                elif rand_val < 0.30:
                    # === TRƯỜNG HỢP 2: IDLE (NGHỈ) ===
                    status = 1 
                    speed = 0
                    d_runtime = 0.0
                    d_heldtime = float(INTERVAL_SECONDS)
                    temp = base_temp + random.uniform(0.5, 2.0) # Mát
                    
                else:
                    # === TRƯỜNG HỢP 3: RUNNING (CHẠY) ===
                    status = 1
                    speed = random.choices([0, 1, 2], weights=[0.2, 0.75, 0.05])[0]
                    d_runtime = float(INTERVAL_SECONDS)
                    d_heldtime = 0.0
                    temp = base_temp + random.uniform(5.0, 10.0) # Ấm
                
                # Cập nhật cộng dồn
                curr_actual += speed
                curr_runtime += d_runtime
                curr_heldtime += d_heldtime
                
                record = {
                    "time": point_time.isoformat(),
                    "DevAddr": dev_id, 
                    "Channel": ch, # Ghi nhận Channel 01 hoặc 02
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

    # 3. Gửi lên Supabase (Gửi 1 cục cho nhanh)
    if all_payloads:
        try:
            # Batch insert giúp giảm số lượng request network
            supabase.table("sensor_data").insert(all_payloads).execute()
            print(f"✅ Đã bơm {len(all_payloads)} điểm dữ liệu (Job hoàn tất)!")
        except Exception as e:
            print(f"❌ Lỗi Insert Supabase: {e}")

# --- ENTRY POINT ---
if __name__ == "__main__":
    # CHỈ CHẠY 1 LẦN RỒI THOÁT (để GitHub Actions báo Success)
    run_worker_batch()