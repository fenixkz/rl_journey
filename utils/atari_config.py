"""
Centralized configuration file for Atari environments in Gymnasium.

This file maps short, user-friendly names to their full environment IDs
and provides standard hyperparameters for training reinforcement learning agents,
including the reward threshold for considering an environment "solved".

The "solved_reward" thresholds are often based on human performance or established
benchmarks in RL literature. They provide a consistent target for evaluating agents.
"""

ATARI_CONFIGS = {
    "adventure": {
        "env_id": "ALE/Adventure-v5",
        "solved_reward": 500.0,  # Highly variable, no standard "solved" score
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "air_raid": {
        "env_id": "ALE/AirRaid-v5",
        "solved_reward": 15000.0,
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "alien": {
        "env_id": "ALE/Alien-v5",
        "solved_reward": 7128.0, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "amidar": {
        "env_id": "ALE/Amidar-v5",
        "solved_reward": 1719.5, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "assault": {
        "env_id": "ALE/Assault-v5",
        "solved_reward": 742.0, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "asterix": {
        "env_id": "ALE/Asterix-v5",
        "solved_reward": 8503.3, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "asteroids": {
        "env_id": "ALE/Asteroids-v5",
        "solved_reward": 47388.7, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "atlantis": {
        "env_id": "ALE/Atlantis-v5",
        "solved_reward": 29028.1, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "bank_heist": {
        "env_id": "ALE/BankHeist-v5",
        "solved_reward": 757.7, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "battle_zone": {
        "env_id": "ALE/BattleZone-v5",
        "solved_reward": 37187.5, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "beam_rider": {
        "env_id": "ALE/BeamRider-v5",
        "solved_reward": 16926.5, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "bowling": {
        "env_id": "ALE/Bowling-v5",
        "solved_reward": 160.7, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "boxing": {
        "env_id": "ALE/Boxing-v5",
        "solved_reward": 12.1, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "breakout": {
        "env_id": "ALE/Breakout-v5",
        "solved_reward": 400.0, # A commonly used threshold for "superhuman" performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "centipede": {
        "env_id": "ALE/Centipede-v5",
        "solved_reward": 12017.0, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "chopper_command": {
        "env_id": "ALE/ChopperCommand-v5",
        "solved_reward": 7387.8, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "crazy_climber": {
        "env_id": "ALE/CrazyClimber-v5",
        "solved_reward": 35829.4, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "demon_attack": {
        "env_id": "ALE/DemonAttack-v5",
        "solved_reward": 1971.0, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "double_dunk": {
        "env_id": "ALE/DoubleDunk-v5",
        "solved_reward": -15.5, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "enduro": {
        "env_id": "ALE/Enduro-v5",
        "solved_reward": 860.5, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "fishing_derby": {
        "env_id": "ALE/FishingDerby-v5",
        "solved_reward": -38.7, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "freeway": {
        "env_id": "ALE/Freeway-v5",
        "solved_reward": 29.6, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "frostbite": {
        "env_id": "ALE/Frostbite-v5",
        "solved_reward": 4534.4, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "gopher": {
        "env_id": "ALE/Gopher-v5",
        "solved_reward": 2412.5, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "gravitar": {
        "env_id": "ALE/Gravitar-v5",
        "solved_reward": 3351.4, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "hero": {
        "env_id": "ALE/Hero-v5",
        "solved_reward": 30993.8,
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "ice_hockey": {
        "env_id": "ALE/IceHockey-v5",
        "solved_reward": 0.9, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "jamesbond": {
        "env_id": "ALE/Jamesbond-v5",
        "solved_reward": 302.8, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "kangaroo": {
        "env_id": "ALE/Kangaroo-v5",
        "solved_reward": 3035.0, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "krull": {
        "env_id": "ALE/Krull-v5",
        "solved_reward": 2665.5, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "kung_fu_master": {
        "env_id": "ALE/KungFuMaster-v5",
        "solved_reward": 22736.3, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "montezuma_revenge": {
        "env_id": "ALE/MontezumaRevenge-v5",
        "solved_reward": 4753.3, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "ms_pacman": {
        "env_id": "ALE/MsPacman-v5",
        "solved_reward": 6951.6, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "name_this_game": {
        "env_id": "ALE/NameThisGame-v5",
        "solved_reward": 8049.0, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "pitfall": {
        "env_id": "ALE/Pitfall-v5",
        "solved_reward": 6463.7, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "pong": {
        "env_id": "ALE/Pong-v5",
        "solved_reward": 20.0, # Win rate of +20 over the built-in AI
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "private_eye": {
        "env_id": "ALE/PrivateEye-v5",
        "solved_reward": 69571.3,
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "qbert": {
        "env_id": "ALE/Qbert-v5",
        "solved_reward": 13455.0, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "riverraid": {
        "env_id": "ALE/Riverraid-v5",
        "solved_reward": 17118.0, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "road_runner": {
        "env_id": "ALE/RoadRunner-v5",
        "solved_reward": 7845.0, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "robotank": {
        "env_id": "ALE/Robotank-v5",
        "solved_reward": 11.9, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "seaquest": {
        "env_id": "ALE/Seaquest-v5",
        "solved_reward": 42054.7, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "skiing": {
        "env_id": "ALE/Skiing-v5",
        "solved_reward": -12682.0,
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "solaris": {
        "env_id": "ALE/Solaris-v5",
        "solved_reward": 12326.7,
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "space_invaders": {
        "env_id": "ALE/SpaceInvaders-v5",
        "solved_reward": 1668.7, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "star_gunner": {
        "env_id": "ALE/StarGunner-v5",
        "solved_reward": 10250.0, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "tennis": {
        "env_id": "ALE/Tennis-v5",
        "solved_reward": -8.0,
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "time_pilot": {
        "env_id": "ALE/TimePilot-v5",
        "solved_reward": 5229.2, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "tutankham": {
        "env_id": "ALE/Tutankham-v5",
        "solved_reward": 167.6, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "up_n_down": {
        "env_id": "ALE/UpNDown-v5",
        "solved_reward": 11693.2, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "venture": {
        "env_id": "ALE/Venture-v5",
        "solved_reward": 1187.5, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "video_pinball": {
        "env_id": "ALE/VideoPinball-v5",
        "solved_reward": 17667.9, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "wizard_of_wor": {
        "env_id": "ALE/WizardOfWor-v5",
        "solved_reward": 4756.5, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "yars_revenge": {
        "env_id": "ALE/YarsRevenge-v5",
        "solved_reward": 54196.4,
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
    "zaxxon": {
        "env_id": "ALE/Zaxxon-v5",
        "solved_reward": 9173.3, # Based on human performance
        "gamma": 0.99,
        "noop_max": 30,
        "frame_skip": 4,
    },
}