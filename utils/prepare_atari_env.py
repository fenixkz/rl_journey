import gymnasium as gym
import ale_py
from ale_py import ALEInterface, LoggerMode
from gymnasium.wrappers import AtariPreprocessing

def get_atari_env(env_id: str) -> gym.Env:
    ALEInterface.setLoggerMode(LoggerMode.Warning)
    env = gym.make(env_id, 
                   frameskip = 1
                   )
    env = AtariPreprocessing(
                            env,
                            noop_max=30,        # To make each episode a bit different
                            frame_skip=4,       # The agent makes a move for next 4 frames, i.e. moves left for 4 frames to speed up training
                            screen_size=84,     # Rescale original image to 84x84
                            grayscale_obs=True, # RGB to GrayScale
                            scale_obs=True,     # Normalize the pixel values from [0-255] to [0-1]
                            terminal_on_life_loss=True, # Restart the episode on the first life loss, instead of waiting for all lifes losses
                        )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4) # Stack last 4 frames as the observation to encode the velocity information
    return env