from mss import mss
import subprocess
import cv2
import numpy as np
import pytesseract
import time
from gym import Env
from gym.spaces import Box, Discrete

class Game(Env):
    def __init__(self):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)
        self.action_space = Discrete(5)

        # setup game capture frame locations
        self.cap = mss()
        self.game_location = self.get_game_location()
        self.done_location = {'top': self.game_location['top'] + 736, 'left': self.game_location['left'] + 220, 'width': 120, 'height': 46}

        self.score = 0
        self.pause_button_sum = 3980

        self.shell = subprocess.Popen(
        ["adb", "shell"],
        stdin=subprocess.PIPE,
        text=True
        )

    def step(self, action):
        # action_map = {
        #     0:'jump',
        #     1:'left',
        #     2:'no_op'
        #     3:'right'
        #     4:'roll'
        # }
        
        
        # press the button for the action given
        self.take_action(action)

        # get the next step
        raw_state, new_state = self.get_observation() 
        
        info = {}
        # check for pause button
        if self.pause_button(raw_state):
            done = False
            # set rewards
            reward = 0.1  # survival reward
            # give reward for staying idle (to remove unnecessary actions)
            if action==2:
                reward += 0.05
            # reward for collecting coins
            reward += 1 if self.is_coin_collected(raw_state) else 0
        else:
            done = True
            reward = 0

        return new_state, reward, done, False, info

    # restart the game
    def reset(self):
        # keep pressing on the screen until the play button comes up
        while True:
            if self.get_done():
                time.sleep(1)
                break
            else:
                self.shell.stdin.write(f"input tap 500 1800\n")
                self.shell.stdin.flush()
                time.sleep(0.5)

        # press the play button at the bottom-right corner of the screen
        self.shell.stdin.write(f"input tap 820 2250\n")
        self.shell.stdin.flush()

        time.sleep(3)

        _, new_state = self.get_observation()
        info = {}
        return new_state, info

    def get_observation(self):
        # capture the screen in the defined area
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3]

        # crop to get only the relevant area
        img = raw[180:600, :]
        # convert image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # resize to reduce resolution
        img = cv2.resize(img, (84,84))
        return raw, img

    def get_done(self):
        # capture the screen in the defined area for the 'PLAY' text
        done_cap = np.array(self.cap.grab(self.done_location))[:,:,:3]

        # convert to grayscale
        done_cap = cv2.cvtColor(done_cap, cv2.COLOR_BGR2GRAY)

        # read the text from the image (apply OCR)
        # custom_config = r'--oem 3 --psm 7'
        res = pytesseract.image_to_string(done_cap).strip()

        # if the text says "PLAY", end the episode
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

        # call pytesseract to read the text using OCR and set to only read numbers
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
        score = pytesseract.image_to_string(img, config=custom_config).strip()

        # if the coin score changes from the last step (coin is collected)
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

        # take the middle row of the image
        sequence = np.array(img[0])
        # calculate the sum of all values to compare with that of the pause button
        sum_pixels = sequence.sum()
        return abs(sum_pixels - self.pause_button_sum) <= 20

    def take_action(self, action):
        # 0 - jump, 1 - left, 3 - right, 4 - roll
        match action:
            case 0:
                # swipe up
                self.shell.stdin.write(f"input swipe 600 1800 600 1000 10\n")
                self.shell.stdin.flush()
                return
            case 1:
                # swipe left
                self.shell.stdin.write(f"input swipe 600 1800 300 1800 10\n")
                self.shell.stdin.flush()
                return
            case 3:
                # swipe right
                self.shell.stdin.write(f"input swipe 300 1800 600 1800 10\n")
                self.shell.stdin.flush()
                return
            case 4:
                # swipe down
                self.shell.stdin.write(f"input swipe 600 1000 600 1800 10\n")
                self.shell.stdin.flush()
                return

    def get_game_location(self):
        # find all window locations
        output = subprocess.check_output(["wmctrl", "-lG"], universal_newlines=True)

        # iterate through each output line to find the scrcpy window
        for line in output.splitlines():
            if "scrcpy" in line:
                # extract x, y, width, height from the output line
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