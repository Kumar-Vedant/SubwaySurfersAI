# import subprocess
# import time
# from mss import mss

# cap = mss()

# def capture_screenshot():
#     subprocess.run(f"adb exec-out screencap -p> screenshot.png", shell=True)

# def ss():
#     cap.grab({'top': 600, 'left': 860, 'width': 400, 'height': 100})

# def tap():
#     subprocess.run(f"adb shell input tap 540 1800", shell=True)

# def swipe():
#     subprocess.run(f"adb shell input swipe 540 1800 540 1000 2", shell=True)

# start = time.time()
# # capture_screenshot()
# # tap()
# swipe()
# end = time.time()

# print(end-start)

import subprocess
import time

def start_persistent_shell():
    """Start a persistent ADB shell process."""
    return subprocess.Popen(
        ["adb", "shell"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1,
    )

def send_command(shell_process, command):
    """Send a command to the persistent ADB shell."""
    shell_process.stdin.write(command + "\n")
    shell_process.stdin.flush()

    output = shell_process.stdout.readline()
    print(f"Shell output: {output.strip()}")  # For debugging
    # time.sleep(0.05)  # Allow the command to execute (adjust if needed)

# def take_screenshot(shell_process, i):
def take_screenshot(shell_process, device_path="/sdcard/screenshot.png", host_path="screenshot.png"):
    # Take the screenshot on the device
    send_command(shell_process, f"screencap -p {device_path}")
    # send_command(shell_process, f"screencap -p > screenshot{i+1}.png")

    # Pull the screenshot to the host machine
    pull_process = subprocess.run(
        ["adb", "pull", device_path, host_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if pull_process.returncode == 0:
        print(f"Screenshot saved to {host_path}")
    else:
        print(f"Failed to pull screenshot: {pull_process.stderr.decode()}")

# Start persistent ADB shell
# shell_process = start_persistent_shell()

# # Take screenshots in a loop (or based on some trigger)
# # for i in range(5):
# #     start = time.time()
# #     take_screenshot(shell_process, host_path=f"screenshot_{i + 1}.png")
# #     # take_screenshot(shell_process, i)
# #     end = time.time()

# #     print(end-start)

# # take_screenshot(shell_process, host_path=f"screenshot.png")
# send_command(shell_process, "input tap 500 1800")

# shell_process.terminate()

# procId = subprocess.Popen(['adb', 'shell'], stdin = subprocess.PIPE)
# procId = subprocess.Popen(
#         ["adb", "shell"],
#         stdin=subprocess.PIPE,
#         # stdout=subprocess.PIPE,
#         # stderr=subprocess.PIPE,
#         # universal_newlines=True,
#         # bufsize=1,
#         text=True
#     )
# # procId.communicate(b'input tap 500 1800')

# start = time.time()
# for i in range(5):
    
#     # procId.stdin.write(f"screencap -p > screenshot_{i}.png\n")
#     procId.stdin.write(f"screencap -p /sdcard/screenshot_{i}.png\n")
#     procId.stdin.flush()


# time.sleep(0.5)

# for i in range(5):
#     subprocess.run(["adb", "pull", f"/sdcard/screenshot_{i}.png", f"screenshot_{i}.png"])

# end = time.time()
# print(end-start)
# procId.stdin.write(b"screencap -p /sdcard/screenshot.png")
# procId.stdin.flush()
# procId.stdin.write(b"input tap 500 1800")
# procId.stdin.flush()

def get_scrcpy_window_location():
    # Run wmctrl to get the list of windows
    output = subprocess.check_output(["wmctrl", "-lG"], universal_newlines=True)

    # Find the scrcpy window
    for line in output.splitlines():
        if "scrcpy" in line.lower():
            parts = line.split()
            x, y, width, height = map(int, parts[2:6])  # Extract x, y, width, height
            return x, y, width, height
        
import numpy as np
import cv2
# from mss import mss

# cap = mss()

# def get_game_location():
#         # find all window locations
#         output = subprocess.check_output(["wmctrl", "-lG"], universal_newlines=True)

#         # iterate through each output line to find the scrcpy window
#         for line in output.splitlines():
#             if "scrcpy" in line:
#                 # extract x, y, width, height from the output line
#                 window_id = line.split()[0]
#                 break

#         # get accurate absolute positions and dimensions using xwininfo
#         xwininfo_output = subprocess.check_output(["xwininfo", "-id", window_id], universal_newlines=True)
#         for line in xwininfo_output.splitlines():
#             if "Absolute upper-left X" in line:
#                 x = int(line.split(":")[1].strip())
#             elif "Absolute upper-left Y" in line:
#                 y = int(line.split(":")[1].strip())
#             elif "Width" in line:
#                 width = int(line.split(":")[1].strip())
#             elif "Height" in line:
#                 height = int(line.split(":")[1].strip())

#         return {'top': y, 'left': x, 'width': width, 'height': height}

# game_location = get_game_location()

# raw = np.array(cap.grab(game_location))[:,:,:3]

# crop to get only the relevant area
# img = raw[180:600, :]
# print(raw.shape)


# loaded_stack = np.load("./expert_data/jump/obs_500.npy")

# print(loaded_stack)
# print(loaded_stack.shape)

# # loaded_stack[0] = np.fliplr(loaded_stack[0])
# # img0 = loaded_stack[0]
# # # # img0 = np.fliplr(img0)

# img = cv2.imshow("Image", loaded_stack[0])
# img = cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# import os

# print(len(os.listdir("./expert_data/right")))

# store_indexes = [len(os.listdir(os.path.join("/home/kumar-vedant/Documents/Development/subwaySurfersAI/expert_data", i))) for i in ["jump", "roll", "left", "right", "no_op"]]

# print(store_indexes)




# loop over all elements in the folder "roll"
# store_index = 21576
# for i in range(21576):
#     loaded_stack = np.load(f"./expert_data/no_op/obs_{i}.npy")

#     # for each element, flip all 4 frames
#     loaded_stack[0] = np.fliplr(loaded_stack[0])
#     loaded_stack[1] = np.fliplr(loaded_stack[1])
#     loaded_stack[2] = np.fliplr(loaded_stack[2])
#     loaded_stack[3] = np.fliplr(loaded_stack[3])

#     # stack the flipped frames
#     image_stack = np.stack([loaded_stack[0], loaded_stack[1], loaded_stack[2], loaded_stack[3]], axis=0)

#     # Save the stack to a .npy file
#     save_path = f"/home/kumar-vedant/Documents/Development/subwaySurfersAI/expert_data/no_op/obs_{store_index}.npy"
#     np.save(save_path, image_stack)
#     store_index += 1

# from sklearn.model_selection import train_test_split
# tr, val = train_test_split(data.label, stratify=data.label, test_size=0.1)

# train_sampler = SubsetRandomSampler(train_indices)
# valid_sampler = SubsetRandomSampler(val_indices)
