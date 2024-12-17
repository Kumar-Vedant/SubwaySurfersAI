from mss import mss
# import pyautogui
import subprocess
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import time
from gym import Env
from gym.spaces import Box, Discrete

import random

class Game(Env):
    def __init__(self):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)
        self.action_space = Discrete(5)

        # setup game capture frame locations
        self.cap = mss()
        # self.game_location = {'top': 670, 'left': 450, 'width': 1000, 'height': 350}
        # self.done_location = {'top': 600, 'left': 860, 'width': 400, 'height': 100}
        self.game_location = self.get_game_location()
        # print(self.game_location)
        self.done_location = {'top': self.game_location['top'] + 736, 'left': self.game_location['left'] + 210, 'width': 130, 'height': 46}

        self.score = 0
        self.pause_button_sum = 3980

        self.shell = subprocess.Popen(
        ["adb", "shell"],
        stdin=subprocess.PIPE,
        # stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
        # universal_newlines=True,
        # bufsize=1,
        text=True
    )

    def step(self, action):
        # action_map = {
        #     0:'jump',
        #     1:'roll',
        #     2:'left'
        #     3:'right'
        #     4:'no_op'
        # }
        if action < 4:
            # press the button for the action given
            # pyautogui.press(action_map[action])
            self.take_action(action)

        # check if game over
        # done, done_cap = self.get_done()
        # get the next step
        raw_state, new_state = self.get_observation() 
        
        info = {}
        # check for pause button
        if self.pause_button(raw_state):
            done = False
            # set rewards
            reward = 1  # survival reward
            reward += 1 if action==4 else 0  # give reward for staying idle (to remove unnecessary actions)
            # reward for collecting coins
            reward += 1 if self.is_coin_collected(raw_state) else 0
        else:
            done = True
            reward = 0

        return new_state, reward, done, False, info
        # return new_state, reward, done

    # restart the game
    def reset(self):
        time.sleep(1)
        # pyautogui.click(x=250, y=250)
        # pyautogui.press('space')

        # keep pressing on the screen until the play button comes up
        while True:
            if self.get_done():
                break
            else:
                self.shell.stdin.write(f"input tap 500 1800\n")
                self.shell.stdin.flush()
                time.sleep(0.5)

        # press the play button at the bottom-right corner of the screen
        self.shell.stdin.write(f"input tap 820 2250\n")
        self.shell.stdin.flush()

        time.sleep(1)

        raw, new_state = self.get_observation()
        info = {}
        return new_state, info

    def get_observation(self):
        # capture the screen in the defined area
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3]
        # convert image to grayscale
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # resize
        resized = cv2.resize(gray, (84,84))
        return raw, resized

    def get_done(self):
        # capture the screen in the defined area for the 'PLAY' text
        done_cap = np.array(self.cap.grab(self.done_location))[:,:,:3]

        # cv2.imshow('image', done_cap)
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()

        # cv2.imwrite("play_test.png", done_cap)
        # done_strings = ['GAME', 'GAHE']
        # done=False

        # convert to grayscale
        done_cap = cv2.cvtColor(done_cap, cv2.COLOR_BGR2GRAY)

        # read the text from the image (apply OCR) and extract the first 4 characters
        # custom_config = r'--oem 3 --psm 7'
        res = pytesseract.image_to_string(done_cap).strip()

        # if the first 4 characters are either "GAME" or "GAHE", end the episode
        # if res in done_strings:
        #     done = True
        return res.lower() == "play"

    def is_coin_collected(self, raw_state):
        # crop the part with the coins
        img = raw_state[75:105, 270:330]

        # convert to grayscale and blur
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.bilateralFilter(img, 9, 75, 75)

        # set thresholds (convert everything above 230 to white and below to black)
        img[img<220] = 0
        img[img>0] = 255

        # invert colors (to make black text on white background)
        img = cv2.bitwise_not(img)

        # call pytesseract to read the text using OCR
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
        score = pytesseract.image_to_string(img, config=custom_config).strip()

        if score != self.score:
            self.score = score
            return True
        else:
            return False
    
    def pause_button(self, raw_state):
        # extract the part with the pause button
        img = raw_state[60:61, 21:42]
        # convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # take the middle row and compare it to pause_button_sequence
        sequence = np.array(img[0])
        sum_pixels = sequence.sum()
        # return sequence == self.pause_button_sequence, sequence, sum_pixels
        return abs(sum_pixels - self.pause_button_sum) <= 20

    def take_action(self, action):
        # 0 - jump, 1 - roll, 2 - left, 3 - right
        match action:
            case 0:
                self.shell.stdin.write(f"input swipe 600 1800 600 1000 10\n")
                self.shell.stdin.flush()
                return
            case 1:
                self.shell.stdin.write(f"input swipe 600 1000 600 1800 10\n")
                self.shell.stdin.flush()
                return
            case 2:
                self.shell.stdin.write(f"input swipe 600 1800 300 1800 10\n")
                self.shell.stdin.flush()
                return
            case 3:
                self.shell.stdin.write(f"input swipe 300 1800 600 1800 10\n")
                self.shell.stdin.flush()
                return

    def get_game_location(self):
        # find all window locations
        output = subprocess.check_output(["wmctrl", "-lG"], universal_newlines=True)

        # iterate through each output line to find the scrcpy window
        for line in output.splitlines():
            if "scrcpy" in line:
                # extract x, y, width, height from the output line
                # parts = line.split()
                window_id = line.split()[0]
                # print(window_id)
                # x, y, width, height = map(int, parts[2:6])
                # print(x, y, width, height)
                # return x, y, width, height
                # return {'top': y, 'left': x, 'width': width, 'height': height}
                break
        # get accurate absolute positions and dimensions using xwininfo
        xwininfo_output = subprocess.check_output(["xwininfo", "-id", window_id], universal_newlines=True)
        # x, y, width, height = None, None, None, None
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

# env = Game()

# state = env.get_observation()

# plt.imshow(cv2.cvtColor(state, cv2.COLOR_BGR2RGB))
# plt.show()

# detect pause button
# img = cv2.imread("ss.png")

# crop = img[40:80, 16:48]
# crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

# print(crop[20])

# cv2.imshow('image', crop)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

# img = cv2.imread("ss.png")

# crop = img[70:110, 270:340]

# crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

# crop[crop<230] = 0
# crop[crop>0] = 255

# # crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# crop = cv2.bitwise_not(crop)

# # image = np.stack((crop, crop, crop), axis=-1)

# # cv2.imwrite("test.png", crop)
# custom_config = r'--oem 3 --psm 7'
# print(pytesseract.image_to_string(crop, config=custom_config))

# cv2.imshow('image', crop)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

# env = Game()

# img = env.get_observation()

# crop = img[75:105, 270:330]

# crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
# # crop = cv2.GaussianBlur(crop, (5, 5), 0)
# # crop = cv2.medianBlur(crop, 5)
# crop = cv2.bilateralFilter(crop, 9, 75, 75)

# crop[crop<230] = 0
# crop[crop>0] = 255

# crop = cv2.bitwise_not(crop)

# custom_config = r'--oem 3 --psm 7'
# print(pytesseract.image_to_string(crop, config=custom_config).strip())

# cv2.imshow('image', crop)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

# img = cv2.imread("play_test.png")

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # custom_config = r'--oem 3 --psm 7'
# # res = pytesseract.image_to_string(img, config=custom_config).strip()
# res = pytesseract.image_to_string(img).strip()

# print(res)

# img, _ = env.get_observation()
# pause_button_sequence = [151, 152, 151, 151, 151, 65, 65, 253, 255, 255, 255, 255, 254, 67, 152, 151, 151, 151, 65, 65, 252, 254, 255, 254, 255, 255, 65, 151, 153, 151, 151, 151]

# img = cv2.imread("ss.png")

# img = img[40:80, 16:48]
# # convert to grayscale
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = img[20]

# # seq = np.asarray(img, dtype=np.uint8)
# seq = img.tolist()

# print(seq == pause_button_sequence)

# print(env.get_done())

# env = Game()

# while True:
#     img, _ = env.get_observation()
#     print(env.is_coin_collected(img))

#     time.sleep(0.1)