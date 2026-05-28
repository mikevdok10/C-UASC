from pymavlink import mavutil
from time import sleep

# Use serial0 for correct Pi UART mapping
master = mavutil.mavlink_connection('/dev/serial0', baud=57600)

print("Waiting for heartbeat...")
master.wait_heartbeat()
print(f"Heartbeat received from system {master.target_system}")

SERVO_CHANNEL = 6
SERVO_HIGH = 2000
SERVO_LOW = 1000

def send_servo(pwm):
    # Send servo command
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        SERVO_CHANNEL,
        pwm,
        0, 0, 0, 0, 0
    )

    print(f"Sent servo command: channel={SERVO_CHANNEL}, pwm={pwm}")

    # Wait for ACK
    ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)

    if ack is None:
        print("❌ No ACK received (command may be ignored)")
        return

    # Check result
    if ack.command == mavutil.mavlink.MAV_CMD_DO_SET_SERVO:
        if ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print("✅ Command ACCEPTED")
        else:
            print(f"❌ Command REJECTED, result code: {ack.result}")
    else:
        print(f"⚠️ Received unrelated ACK: {ack.command}")

while True:
    input("Press Enter to toggle servo...")

    send_servo(SERVO_HIGH)
    sleep(2)

    send_servo(SERVO_LOW)
    sleep(2)