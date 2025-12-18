import os
import random
import time
import pandas as pd
from datetime import datetime, timedelta
from supabase import create_client
import openmeteo_requests
import requests_cache
from retry_requests import retry

print("🤖 IOT WORKER: Bắt đầu bơm dữ liệu CHUẨN (Matched with Training Data)...")

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
    # --- CẤU HÌNH QUAN TRỌNG ĐỂ KHỚP MODEL ---
    # Model được train với dữ liệu ~20s/mẫu, nên worker phải sinh ra tương tự
    INTERVAL_SECONDS = 20  
    
    # Sinh dữ liệu cho 20 phút (60 điểm * 20s = 1200s = 20 phút)
    POINTS_PER_RUN = 60    
    
    base_temp, base_hum = get_weather()
    all_payloads = []
    
    # Lùi thời gian lại để bơm dữ liệu quá khứ gần
    start_time_base = datetime.now() - timedelta(seconds=POINTS_PER_RUN * INTERVAL_SECONDS)

    for dev in DEVICES:
        dev_id = dev['id']
        ch = dev['ch']
        
        # 1. Lấy trạng thái cũ từ DB để cộng dồn tiếp
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
            
            # --- LOGIC MÔ PHỎNG CHUẨN ---
            
            # Xác định trạng thái máy: 95% là chạy (Status 1), 5% là dừng (Status 2)
            is_running = random.random() < 0.95 
            
            if is_running:
                status = 1
                # Khi chạy: Speed là số sản phẩm làm được trong 20s.
                # Thường là 1 sp, thỉnh thoảng 0 (chưa xong), hiếm khi 2 (làm nhanh)
                speed = random.choices([0, 1, 2], weights=[0.2, 0.75, 0.05])[0]
                
                # Delta thời gian
                d_runtime = float(INTERVAL_SECONDS)
                d_heldtime = 0.0
                
                # Nhiệt độ máy khi chạy sẽ nóng hơn môi trường khoảng 5-8 độ
                temp = base_temp + random.uniform(5.0, 8.0)
                
            else:
                status = 2
                # Khi dừng: Speed chắc chắn là 0
                speed = 0
                
                # Delta thời gian
                d_runtime = 0.0
                d_heldtime = float(INTERVAL_SECONDS)
                
                # Nhiệt độ máy khi dừng sẽ nguội dần (gần bằng môi trường)
                temp = base_temp + random.uniform(0.5, 2.0)
            
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
                "Speed": float(speed),          # Quan trọng: Speed giờ là 0, 1 hoặc 2
                "d_RunTime": d_runtime,         # Quan trọng: 20.0 hoặc 0.0
                "d_HeldTime": d_heldtime,       # Quan trọng: 0.0 hoặc 20.0
                "Temp": float(f"{temp:.2f}"), 
                "Humidity": base_hum
            }
            all_payloads.append(record)

    # 3. Gửi lên Supabase
    if all_payloads:
        try:
            # Gửi từng batch nhỏ để tránh quá tải nếu cần, ở đây gửi hết
            supabase.table("sensor_data").insert(all_payloads).execute()
            print(f"✅ Đã bơm {len(all_payloads)} điểm dữ liệu CHUẨN (Speed 0-2, Interval 20s)!")
        except Exception as e:
            print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    run_worker_batch()