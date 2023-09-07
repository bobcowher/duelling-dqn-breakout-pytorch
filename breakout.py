import collections
import cv2
import gym
import numpy as np
from PIL import Image
import torch



class DQNBreakout(gym.Wrapper):

    def __init__(self, render_mode='rgb_array', repeat=4, clip_reward=False, no_ops=0,
                 fire_first=False, device='cpu'):
        env = gym.make("BreakoutNoFrameskip-v4", render_mode=render_mode)

        super(DQNBreakout, self).__init__(env)

        self.repeat = repeat
        self.image_shape = (84,84)

        self.frame_buffer = []
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first
        self.device = device

    def step(self, action):
        total_reward = 0
        done = False

        # We're repeating for 4 actions to speed up training and
        # averaging over the last 2 to remove stutter in the Atari library.
        for i in range(self.repeat):
            observation, reward, done, truncated, info = self.env.step(action)
            total_reward += reward

            self.frame_buffer.append(observation)

            if done:
                break

        max_frame = np.max(self.frame_buffer[-2:], axis=0)
        max_frame = self.process_observation(max_frame)
        max_frame = max_frame.to(self.device)

        total_reward = torch.tensor(total_reward).view(1, -1).float()
        total_reward = total_reward.to(self.device)

        done = torch.tensor(done).view(1, -1)
        done = done.to(self.device)

        return max_frame, total_reward, done, info

    def reset(self):
        self.frame_buffer = []
        observation, _ = self.env.reset()
        observation = self.process_observation(observation)

        return observation

    def process_observation(self, observation):

        img = Image.fromarray(observation)
        img = img.resize(self.image_shape)
        img = img.convert("L")
        img = np.array(img)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.unsqueeze(0)
        img = img / 255.0

        img = img.to(self.device)

        return img






# class PreprocessFrame(gym.ObservationWrapper):
    #
    #
    #
    #
    # def reset(self):
    #     obs = self.env.reset()
    #     no_ops = np.random.randint(self.no_ops)+1 if self.no_ops > 0 else 0
    #
    #     for _ in range(no_ops):
    #         _, _, done, _ = self.env.step(0)
    #         if done:
    #             self.env.reset()
    #
    #     if self.fire_first:
    #         assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
    #         obs, _, _, _ = self.env.step(1)
    #
    #
    #     self.frame_buffer = []
    #     self.frame_buffer[0] = obs
    #
    #     return obs

# class StackFrames(gym.ObservationWrapper):
#     def __init__(self, env, repeat):
#         super(StackFrames, self).__init__(env)
#         self.observation_space = gym.spaces.Box(
#             env.observation_space.low.repeat(repeat, axis=0),
#             env.observation_space.high.repeat(repeat, axis=0),
#             dtype=np.float32
#         )
#         self.stack - collections.deque(maxlen=repeat)
#
#     def reset(self):
#         self.stack.clear()
#         observation = self.env.reset()
#         for _ in range(self.stack.maxlen):
#             self.stack.append(observation)
#
#         return np.array(self.stack).reshape(self.observation_space.low.shape)
#
#     def observation(self, observation):
#         self.stack.append(observation)
#
#         return np.array(self.stack).reshape(self.observation_space.low.shape)




