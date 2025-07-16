from gym import Wrapper
from gym.wrappers import FrameStack

class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        # total_reward = 0.0
        done = False

        next_state, reward, done, trunc, info = self.env.step(action)
        # total_reward += reward

        # take no action for the next n-1(skip) steps
        for _ in range(self.skip-1):
            # take the step
            next_state, reward, done, trunc, info = self.env.step(2)
            # total_reward += reward
            if done:
                break
        return next_state, reward, done, trunc, info

# apply wrappers - SkipFrame and FrameStack
def apply_wrappers(env):
    # skip 4 frames (don't take any action for the next 3 frames)
    env = SkipFrame(env, skip=6)
    # stack 4 frames to give a sense of motion
    env = FrameStack(env, num_stack=4, lz4_compress=True)

    return env