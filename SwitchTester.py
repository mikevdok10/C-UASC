from pymavlink import mavutil
import time 

connectionString = '/dev/ttyAMA0'
baudRate = 57600

print("Connecting to Pixhawk...")

master = mavutil.mavlink_connection(connectionString, baud=baudRate)
master.wait_heartbeat()
print("Connected to Pixhawk!")

master.mav.request_data_stream_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS, 10,1
)

print("\nMove switches on trasnmitter\n")

while True:
    msg = master.recv_match(type='RC_CHANNELS', blocking=True)

    if not msg:
        continue
    print(
        f"CH1: {msg.chan1_raw}, CH2: {msg.chan2_raw}, CH3: {msg.chan3_raw}, CH4: {msg.chan4_raw}, CH5: {msg.chan5_raw}, CH6: {msg.chan6_raw}, CH7: {msg.chan7_raw}, CH8: {msg.chan8_raw}"
    )

    time.sleep(0.05)