# Subway Surfers AI

A CNN-based Reinforcement Learning agent to play Subway Surfers directly on an Android device through USB.

## Demo

Here's the AI playing the game at different training stages:
<p align="center">
  <img src="demo/demo.gif"/>
</p>

# Features

- Real-time data capture (30FPS, 400ms latency) using scrcpy for screen mirroring and ADB for direct control.
- Custom Implementation of Proximal Policy Optimization (PPO) and Deep Q-Network (DQN) in PyTorch.
- Script to capture human-played data for Behavioral Cloning to reduce training time.
- Uses OpenAI's Gymnasium framework for environment setup.
- Coin collection detection for reward framing.
- Automated detection and implementation of Game Over and restart using OpenCV and pytessaract for OCR.
- Live plotting of rewards over episodes.

## Usage

```bash
git clone https://github.com/Kumar-Vedant/SubwaySurfersAI.git
```

1. Enable Developer options on your Android Device.
2. Turn On USB Debugging from Developer Options.
3. Connect your phone to the PC using USB.
4. Open a terminal and start screen mirroring using `bash scrcpy --max-size 800 --max-fps 30 --no-audio --window-title "scrcpy"`. A window should pop-up with the name scrcpy which mirrors your Android device's screen.
5. Open another terminal and run the main.py script of either the DQN or PPO folders. If training, set the TRAINING variable to True. If you only want to run inference on a pretrained model, set it to False and put the model's path in the else part of the check on TRAINING.
6. Another window should pop-up with a graph that plots rewards vs episodes.
7. The game should run automatically, restarting when it dies.

If collecting human-played data for Behavioral Cloning:

1. Enable Developer options on your Android Device.
2. Turn On USB Debugging from Developer Options.
3. Connect your phone to the PC using USB.
4. Open a terminal and start screen mirroring using `bash scrcpy --max-size 800 --max-fps 30 --no-audio --window-title "scrcpy"`. A window should pop-up with the name scrcpy which mirrors your Android device's screen.
5. Open another terminal and run the data_recorder.py
6. It will create folders for each action (jump, left, no_op, right, roll) in expert_data folder.
7. Use arrow keys for actions and space bar for no_op and the script will take that action directly on the phone and store screenshots in the corresponding folders.

## Tech Stack

- AI Model: Python, PyTorch, OpenCV, pytessaract
- Environment: OpenAI Gymnasium
- Android Interface: scrcpy, ADB


## License

```
MIT License

Copyright (c) 2025 Kumar Vedant

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
