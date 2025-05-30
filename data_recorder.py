from mss import mss
import subprocess
import cv2
import numpy as np
import os

game_frames = []
cap = mss()
# start with the previous index to continue dataset without overwriting
store_indexes = [len(os.listdir(os.path.join("/home/kumar-vedant/Documents/Development/subwaySurfersAI/expert_data", i))) for i in ["jump", "roll", "left", "right", "no_op"]]

# find all window locations
def get_game_location():
    output = subprocess.check_output(["wmctrl", "-lG"], universal_newlines=True)

    # iterate through each output line to find the scrcpy window
    for line in output.splitlines():
        if "scrcpy" in line:
            # extract the windowID from the output line
            window_id = line.split()[0]
            break

    # get accurate absolute positions and dimensions using xwininfo
    xwininfo_output = subprocess.check_output(["xwininfo", "-id", window_id], universal_newlines=True)
    for line in xwininfo_output.splitlines():
        if "Absolute upper-left X" in line:
            x = int(line.split(":")[1].strip())
        elif "Absolute upper-left Y" in line:
            y = int(line.split(":")[1].strip())
        elif "Width" in line:
            width = int(line.split(":")[1].strip())
        elif "Height" in line:
            height = int(line.split(":")[1].strip())

    return {'top': y, 'left': x, 'width': width, 'height': height}

def capture_screen():
    # capture the screen in the defined area
    raw = np.array(cap.grab(game_location))[:,:,:3]

    # crop to get only the relevant area
    img = raw[180:600, :]
    # convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize to reduce resolution
    img = cv2.resize(img, (84,84))

    return raw, img

game_location = get_game_location()
# game_location = {'top': 670, 'left': 450, 'width': 1000, 'height': 350}

from pynput import keyboard

# if an action is taken in the current state, store the 4 frames in the folder of the action
def on_press(key):
    # check for arrow keys
    if key == keyboard.Key.up:
        print("UP")
        
        shell.stdin.write(f"input swipe 600 1800 600 1000 10\n")
        shell.stdin.flush()

        image_stack = np.stack([game_frames[0], game_frames[1], game_frames[2], game_frames[3]], axis=0)

        # Save the stack to a .npy file
        save_path = f"/home/kumar-vedant/Documents/Development/subwaySurfersAI/expert_data/jump/obs_{store_indexes[0]}.npy"
        np.save(save_path, image_stack)
        store_indexes[0] += 1

    elif key == keyboard.Key.down:
        print("DOWN")

        shell.stdin.write(f"input swipe 600 1000 600 1800 10\n")
        shell.stdin.flush()

        image_stack = np.stack([game_frames[0], game_frames[1], game_frames[2], game_frames[3]], axis=0)

        # Save the stack to a .npy file
        save_path = f"/home/kumar-vedant/Documents/Development/subwaySurfersAI/expert_data/roll/obs_{store_indexes[1]}.npy"
        np.save(save_path, image_stack)
        store_indexes[1] += 1

    elif key == keyboard.Key.left:
        print("LEFT")

        shell.stdin.write(f"input swipe 600 1800 300 1800 10\n")
        shell.stdin.flush()

        image_stack = np.stack([game_frames[0], game_frames[1], game_frames[2], game_frames[3]], axis=0)

        # Save the stack to a .npy file
        save_path = f"/home/kumar-vedant/Documents/Development/subwaySurfersAI/expert_data/left/obs_{store_indexes[2]}.npy"
        np.save(save_path, image_stack)
        store_indexes[2] += 1

    elif key == keyboard.Key.right:
        print("RIGHT")

        shell.stdin.write(f"input swipe 300 1800 600 1800 10\n")
        shell.stdin.flush()

        image_stack = np.stack([game_frames[0], game_frames[1], game_frames[2], game_frames[3]], axis=0)

        # Save the stack to a .npy file
        save_path = f"/home/kumar-vedant/Documents/Development/subwaySurfersAI/expert_data/right/obs_{store_indexes[3]}.npy"
        np.save(save_path, image_stack)
        store_indexes[3] += 1

    elif key == keyboard.Key.space:
        print("NO_OP")

        image_stack = np.stack([game_frames[0], game_frames[1], game_frames[2], game_frames[3]], axis=0)

        # Save the stack to a .npy file
        save_path = f"/home/kumar-vedant/Documents/Development/subwaySurfersAI/expert_data/no_op/obs_{store_indexes[4]}.npy"
        np.save(save_path, image_stack)
        store_indexes[4] += 1



shell = subprocess.Popen(
        ["adb", "shell"],
        stdin=subprocess.PIPE,
        text=True)

# key press listener setup
listener = keyboard.Listener(on_press=on_press)
listener.start()

while True:
    # capture current state screenshot
    raw, img = capture_screen()

    # push the new state
    game_frames.append(img)
    # remove the earliest state if more than 4 are stored
    if len(game_frames)>4:
        game_frames.pop(0)