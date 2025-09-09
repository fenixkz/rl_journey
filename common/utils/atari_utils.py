import ale_py  # noqa: F401, it is a special case
import gymnasium as gym
import numpy as np
from ale_py import ALEInterface, LoggerMode
from gymnasium.wrappers import AtariPreprocessing


class FireResetEnv(gym.Wrapper):
    """
    A wrapper for single Atari environments that takes the FIRE action on reset.
    This is intended to be used on individual environments before vectorization.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        action_meanings = env.unwrapped.get_action_meanings()

        if "FIRE" not in action_meanings:
            raise ValueError(f"Environment {env.spec.id} does not have a FIRE action.")

        self._fire_action = action_meanings.index("FIRE")

    def reset(self, **kwargs):
        """
        Resets the environment and performs the initial FIRE action.
        """
        # First, reset the underlying environment to get a fresh state
        obs, info = self.env.reset(**kwargs)

        # Now, take the FIRE action to start the game
        obs, _, _, _, _ = self.env.step(self._fire_action)

        return obs, info


def get_atari_env(env_id: str, terminal_on_life: bool = True) -> gym.Env:
    ALEInterface.setLoggerMode(LoggerMode.Warning)
    env = gym.make(env_id, frameskip=1, render_mode="rgb_array")
    env = AtariPreprocessing(
        env,
        noop_max=30,  # To make each episode a bit different
        frame_skip=4,  # The agent makes a move for next 4 frames, i.e. moves left for 4 frames to speed up training
        screen_size=84,  # Rescale original image to 84x84
        grayscale_obs=True,  # RGB to GrayScale
        scale_obs=True,  # Normalize the pixel values from [0-255] to [0-1]
        terminal_on_life_loss=terminal_on_life,  # Finish the episode on the first life loss
    )
    env = gym.wrappers.FrameStackObservation(
        env, stack_size=4
    )  # Stack last 4 frames as the observation to encode the velocity information
    env = FireResetEnv(env)  # For auto-fire
    return env


def clip_reward(reward: float) -> float:
    return np.sign(reward)
