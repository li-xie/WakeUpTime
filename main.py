import asyncio
import math
import pickle
import os
import sys
from signal import SIGINT
import subprocess
from time import sleep
from datetime import datetime

from bleak import BleakClient
from bleak.uuids import uuid16_dict


""" Predefined UUID (Universal Unique Identifier) mapping are based on Heart Rate GATT service Protocol that most
Fitness/Heart Rate device manufacturer follow (Polar H10 in this case) to obtain a specific response input from
the device acting as an API """

uuid16_dict = {v: k for k, v in uuid16_dict.items()}

# This is the device MAC ID, please update with your device ID
ADDRESS = "EC:CD:5B:B7:75:F5"

## UUID for model number ##
MODEL_NBR_UUID = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(
    uuid16_dict.get("Model Number String")
)


## UUID for manufacturer name ##
MANUFACTURER_NAME_UUID = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(
    uuid16_dict.get("Manufacturer Name String")
)

## UUID for battery level ##
BATTERY_LEVEL_UUID = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(
    uuid16_dict.get("Battery Level")
)

## UUID for connection establsihment with device ##
PMD_SERVICE = "FB005C80-02E7-F387-1CAD-8ACD2D8DF0C8"

## UUID for Request of stream settings ##
PMD_CONTROL = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8"

## UUID for Request of start stream ##
PMD_DATA = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"

## UUID for Request of ECG Stream ##
ECG_WRITE = bytearray([0x02, 0x00, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0E, 0x00])

## For Plolar H10  sampling frequency ##
ECG_SAMPLING_FREQ = 130


ecg_session_data = []
ecg_session_time = []
record_flag = True

# Keyboard Interrupt Handler


def handler():
    global record_flag
    record_flag = False
    # loop = asyncio.get_running_loop()
    # for task in asyncio.all_tasks(loop=loop):
    #     task.cancel()
    print("  key board interrupt received...")
    print("----------------Recording stopped------------------------")


# Bit conversion of the Hexadecimal stream
def data_conv(sender, data):
    if data[0] == 0x00:
        timestamp = convert_to_unsigned_long(data, 1, 8)
        step = 3
        samples = data[10:]
        offset = 0
        while offset < len(samples):
            ecg = convert_array_to_signed_int(samples, offset, step)
            offset += step
            ecg_session_data.extend([ecg])
            ecg_session_time.extend([0])
        ecg_session_time[-1] = timestamp


def convert_array_to_signed_int(data, offset, length):
    return int.from_bytes(
        bytearray(data[offset: offset + length]), byteorder="little", signed=True,
    )


def convert_to_unsigned_long(data, offset, length):
    return int.from_bytes(
        bytearray(data[offset: offset + length]), byteorder="little", signed=False,
    )
    # print(f'run: {ecg_session_time[-1]}')


async def main(dir_name, wake_str):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    max_wake = [int(x) for x in wake_str.split(":")]
    time_convert = lambda x: x[0]*60+x[1]
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(SIGINT, handler)
    async with BleakClient(ADDRESS) as client:
        if client.is_connected:
            print("---------Device connected--------------")

        model_number = await client.read_gatt_char(MODEL_NBR_UUID)
        print("Model Number: {0}".format("".join(map(chr, model_number))))

        att_read = await client.read_gatt_char(PMD_CONTROL)

        await client.write_gatt_char(PMD_CONTROL, ECG_WRITE)

        # ECG stream started
        await client.start_notify(PMD_DATA, data_conv)

        print("Collecting ECG data...")

        i = 0
        process_output = None
        # try:
        while record_flag:
            # if (time_convert(now_list)>time_convert(max_wake)): #(now_list[0] <22) &
            #     break
            # Collecting ECG data
            await asyncio.sleep(60)
            i += 1
            global ecg_session_time, ecg_session_data
            ecg_time_write = ecg_session_time
            ecg_session_time = []
            ecg_data_write = ecg_session_data
            ecg_session_data = []
            with open(dir_name+f'/t{i}.pkl', "wb") as t_file:
                pickle.dump(ecg_time_write, t_file)
            with open(dir_name+f'/d{i}.pkl', "wb") as d_file:
                pickle.dump(ecg_data_write, d_file)
            if i == 40:
                data_process = subprocess.Popen(["python", "dynamic_analysis", dir_name, wake_str], \
                                                stdout=subprocess.PIPE)
            if i>40:
                poll = data_process.poll()
                if poll != None:
                    process_output = data_process.stdout.read1().decode('utf-8').strip()
                    process_error = data_process.stderr.read1().decode('utf-8').strip()
                    record_flag = False
                    break
        # Stop the stream once data is collected
        await client.stop_notify(PMD_DATA)
        print("Stopping ECG data...")
        print("[CLOSED] application closed.")
    if process_output != 'wake':
        sleep_flag = True
        while sleep_flag:
            now = datetime.now()
            now_list = [now.hour, now.minute]
            if (time_convert(now_list)>time_convert(max_wake)): #(now_list[0] <22) &
                sleep_flag = False
                break
            else:
                sleep(60)
    play_process = subprocess.run(["ffplay", "sample.mp3"])



if __name__ == "__main__":
    dir_name = sys.argv[1]
    wake_str = sys.argv[2]
    os.environ["PYTHONASYNCIODEBUG"] = str(1)
    asyncio.run(main(dir_name, wake_str))
