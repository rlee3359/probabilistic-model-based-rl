import gym
import numpy as np
import cv2
import copy

class Env:
  def __init__(self, vision=False):
    self.vision = vision
    self.W = 400
    self.ACT_SCALE = 0.01
    self.TL = 100

    self.action_space = gym.spaces.Box(low=-1,high=1, shape=[3])
    if not self.vision:
      self.observation_space = gym.spaces.Box(low=-1,high=1, shape=[6])
    else:
      self.observation_space = gym.spaces.Box(low=0.0, high=1, shape=[84,84,3])

    self.goal  = np.array([0.5,0.5,0.5,0.0,0.0,0.0])
    self.state = np.array([0.5,0.5,0.5])

    self.reset()
    

  def reset(self):
    self.img = np.zeros((self.W, self.W, 3), np.uint8)
    self.img[:] = (80,80,80)

    # self.goal  = np.random.uniform(0, 1, (3))
    self.state = np.random.uniform(0, 1, (3))
    self.prev_state = self.state
    self.ts = 0
    return self._get_state()

  def step(self, action):
    self.state = np.clip(self.state + np.clip(action, -1, 1)*self.ACT_SCALE, 0, 1)
    self.ts += 1
    reward = self._reward()
    done = self._done()
    return self._get_state(), reward, done, {}
  
  def _get_state(self):
    if not self.vision:
      vel = self.state - self.prev_state
      state = np.hstack([self.state, vel])
      self.prev_state = self.state
    else:
      state = copy.deepcopy(self.get_image(self.state))
      state = cv2.resize(state, (84,84))
    return state

  def _reward(self):
    return -np.sum((self.goal[:3] - self.state)**2)

  def _done(self):
    outside = np.any(self.state > 1) or np.any(self.state < 0)
    return self.ts >= self.TL or outside
  

  def get_image(self, state_list, img = None, iteration=0):
    if img is None:
      img = np.zeros((self.W, self.W, 3), np.uint8)
      img[:] = (150,150,150)

    # Convert state to pixels
    def to_pix(state):
      state = copy.deepcopy(state) * self.W
      return int(state[0]), int(state[1])
    def depth(state):
      return int(state[2]*5+1)

    cv2.circle(img, to_pix(self.goal),  depth(self.goal),  (100,100,200), -1, lineType=cv2.LINE_AA)
    for i,state in enumerate(state_list):
      color = (120-i*1, iteration*10,100-iteration*10)
      cv2.circle(img, to_pix(state), depth(state), color, -1, lineType=cv2.LINE_AA)

    # self.img = cv2.addWeighted(self.img, 0.8, img, 0.2, 0)
    return img


  def render(self, state_list=None, trajectories=[], iteration=0, img = None):
    if state_list is None:
      # img = None
      for state_list in trajectories:
        img = self.get_image(state_list, img, iteration=iteration)
      cv2.imshow("Render", img)
      cv2.waitKey(10)
    else:
      img = self.get_image(state_list, iteration=iteration)
    cv2.namedWindow("Render", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Render", img)
    cv2.waitKey(10)
    return img


# env = Env()
# obs = env.reset()
# done = False
# while not done:
#   act = env.action_space.sample()
#   env.step(act)
#   env.render()
#

