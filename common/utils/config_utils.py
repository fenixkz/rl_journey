from common.utils.configs import ATARI_CONFIGS, CONFIGS, CONTINUOUS_CONFIGS


def get_env_config(env_name: str, include_atari: bool = True) -> dict:
    """
    Retrieves the configuration for a given environment name.

    Args:
        env_name: The short name of the environment (e.g., "pong").
        include_atari: Whether or not include atari envs in the search

    Returns:
        A dictionary containing the environment's configuration.

    Raises:
        ValueError: If the short name is not found in the configs.
    """

    if env_name.lower() in CONFIGS:
        return CONFIGS[env_name.lower()]
    if include_atari and env_name.lower() in ATARI_CONFIGS:
        return ATARI_CONFIGS[env_name.lower()]
    raise ValueError(f"Unknown environment: '{env_name}'. Please, double check the name.")


def get_cont_env_config(env_name: str) -> dict:
    if env_name.lower() in CONTINUOUS_CONFIGS:
        return CONTINUOUS_CONFIGS[env_name.lower()]
    raise ValueError(f"Unknown environment: '{env_name}'. Please, double check the name.")
