CONFIGS = {
    "cartpole": {
        "env_id": "CartPole-v1",
        "solved_reward": 450.0,  # Official is 495, but relax for experiments
        "description": "Classic control - balance pole on cart",
        "min_reward": 0.0,  # Episode terminates, so minimum is 0
        "max_reward": 500.0,  # Maximum episode length
    },
    "mountaincar": {
        "env_id": "MountainCar-v0",
        "solved_reward": -110.0,  # MountainCar has negative rewards
        "description": "Get car to top of mountain using momentum",
        "min_reward": -200.0,  # Maximum episode length with -1 per step
        "max_reward": 0.0,  # Reward when reaching goal
    },
    "lunarlander": {
        "env_id": "LunarLander-v3",
        "solved_reward": 200.0,
        "description": "Land spacecraft safely on moon surface",
        "min_reward": -300.0,  # Crashing and fuel consumption
        "max_reward": 300.0,  # Perfect landing with bonuses
    },
    "acrobot": {
        "env_id": "Acrobot-v1",
        "solved_reward": -100.0,
        "description": "Swing up underactuated pendulum",
        "min_reward": -500.0,  # Maximum episode length with -1 per step
        "max_reward": 0.0,  # Reward when reaching goal
    },
    "taxi": {
        "env_id": "Taxi-v3",
        "solved_reward": 9.0,
        "description": "Train a self-driving taxi",
        "min_reward": -200.0,  # Maximum episode length with penalties
        "max_reward": 20.0,  # Successful delivery reward
    },
}

CONTINUOUS_CONFIGS = {
    "pendulum": {
        "env_id": "Pendulum-v1",
        "solved_reward": -150.0,
        "description": "Swing up an inverted pendulum and keep it upright",
        "min_reward": -1800.0,  # Worst possible performance over 200 steps
        "max_reward": 0.0,  # Perfect performance (zero cost)
    },
    "mountaincar_continuous": {
        "env_id": "MountainCarContinuous-v0",
        "solved_reward": 90.0,
        "description": "Get car to top of mountain using momentum (continuous action)",
        "min_reward": -100.0,  # Failing to reach the goal
        "max_reward": 105.0,  # Reaching the goal very efficiently
    },
    "lunarlander_continuous": {
        "env_id": "LunarLander-v3",  # Note: gym.make("LunarLander-v3", continuous=True)
        "solved_reward": 200.0,
        "description": "Land spacecraft safely on moon surface (continuous control)",
        "min_reward": -300.0,  # Crashing and fuel consumption
        "max_reward": 300.0,  # Perfect landing with bonuses
    },
    "bipedalwalker": {
        "env_id": "BipedalWalker-v3",
        "solved_reward": 300.0,
        "description": "Teach a 2D bipedal robot to walk",
        "min_reward": -150.0,  # Crashing early
        "max_reward": 350.0,  # Very fast and stable walking
    },
    "bipedalwalker_hardcore": {
        "env_id": "BipedalWalkerHardcore-v3",
        "solved_reward": 300.0,
        "description": "Teach a 2D bipedal robot to walk over difficult terrain",
        "min_reward": -200.0,  # Crashing or stumbling on obstacles
        "max_reward": 350.0,  # Very efficient run over all obstacles
    },
    # --- MuJoCo Environments ---
    "halfcheetah": {
        "env_id": "HalfCheetah-v5",
        "solved_reward": 8000.0,  # No official threshold, benchmark for high score
        "description": "Make a 2D cheetah-like robot run as fast as possible",
        "min_reward": -500.0,  # Falling over and incurring control costs
        "max_reward": 16000.0,  # Representative of top-performing agents
    },
    "hopper": {
        "env_id": "Hopper-v5",
        "solved_reward": 3500.0,
        "description": "Make a one-legged robot hop forward without falling",
        "min_reward": -50.0,  # Falling over immediately
        "max_reward": 4000.0,  # A very good hopping performance
    },
    "walker2d": {
        "env_id": "Walker2d-v5",
        "solved_reward": 4500.0,
        "description": "Teach a 2D bipedal robot to walk forward",
        "min_reward": -50.0,  # Falling over immediately
        "max_reward": 6000.0,  # A very good walking performance
    },
    "ant": {
        "env_id": "Ant-v5",
        "solved_reward": 6000.0,
        "description": "Teach a four-legged 'ant' robot to walk forward",
        "min_reward": -100.0,  # Flipping over and incurring costs
        "max_reward": 8000.0,  # A strong walking performance
    },
    "humanoid": {
        "env_id": "Humanoid-v5",
        "solved_reward": 6000.0,
        "description": "Make a 3D simulated human model walk forward",
        "min_reward": 0.0,  # Falling over immediately
        "max_reward": 10000.0,  # A very strong, stable walking performance
    },
}

ATARI_CONFIGS = {
    # --- Special case for Pong with tuned hyperparameters ---
    "pong": {
        "env_id": "ALE/Pong-v5",
        "solved_reward": 20.0,  # Win rate of +20 over the built-in AI
        "min_reward": -21.0,  # Maximum loss score
        "max_reward": 21.0,  # Maximum win score
    },
    "breakout": {
        "env_id": "ALE/Breakout-v5",
        "solved_reward": 400.0,
        "min_reward": 0.0,  # No negative rewards
        "max_reward": 500.0,  # Typical high score range
    },
    "space_invaders": {
        "env_id": "ALE/SpaceInvaders-v5",
        "solved_reward": 1668.7,
        "min_reward": 0.0,  # No negative rewards
        "max_reward": 3000.0,  # High score potential
    },
    "seaquest": {
        "env_id": "ALE/Seaquest-v5",
        "solved_reward": 42054.7,
        "min_reward": 0.0,  # No negative rewards
        "max_reward": 50000.0,  # High score potential
    },
    "beam_rider": {
        "env_id": "ALE/BeamRider-v5",
        "solved_reward": 16926.5,
        "min_reward": 0.0,  # No negative rewards
        "max_reward": 25000.0,  # High score potential
    },
    "enduro": {
        "env_id": "ALE/Enduro-v5",
        "solved_reward": 860.5,
        "min_reward": 0.0,  # No negative rewards
        "max_reward": 1500.0,  # High score potential
    },
    "qbert": {
        "env_id": "ALE/Qbert-v5",
        "solved_reward": 13455.0,
        "min_reward": 0.0,  # No negative rewards
        "max_reward": 20000.0,  # High score potential
    },
    "ms_pacman": {
        "env_id": "ALE/MsPacman-v5",
        "solved_reward": 6951.6,
        "min_reward": 0.0,  # No negative rewards
        "max_reward": 10000.0,  # High score potential
    },
    "asteroids": {
        "env_id": "ALE/Asteroids-v5",
        "solved_reward": 47388.7,
        "min_reward": 0.0,  # No negative rewards
        "max_reward": 60000.0,  # High score potential
    },
    "adventure": {
        "env_id": "ALE/Adventure-v5",
        "solved_reward": 500.0,
        "min_reward": 0.0,  # No negative rewards
        "max_reward": 1000.0,  # High score potential
    },
    "air_raid": {
        "env_id": "ALE/AirRaid-v5",
        "solved_reward": 15000.0,
        "min_reward": 0.0,
        "max_reward": 20000.0,
    },
    "alien": {
        "env_id": "ALE/Alien-v5",
        "solved_reward": 7128.0,
        "min_reward": 0.0,
        "max_reward": 10000.0,
    },
    "amidar": {
        "env_id": "ALE/Amidar-v5",
        "solved_reward": 1719.5,
        "min_reward": 0.0,
        "max_reward": 3000.0,
    },
    "assault": {
        "env_id": "ALE/Assault-v5",
        "solved_reward": 742.0,
        "min_reward": 0.0,
        "max_reward": 1500.0,
    },
    "asterix": {
        "env_id": "ALE/Asterix-v5",
        "solved_reward": 8503.3,
        "min_reward": 0.0,
        "max_reward": 12000.0,
    },
    "atlantis": {
        "env_id": "ALE/Atlantis-v5",
        "solved_reward": 29028.1,
        "min_reward": 0.0,
        "max_reward": 40000.0,
    },
    "bank_heist": {
        "env_id": "ALE/BankHeist-v5",
        "solved_reward": 757.7,
        "min_reward": 0.0,
        "max_reward": 1500.0,
    },
    "battle_zone": {
        "env_id": "ALE/BattleZone-v5",
        "solved_reward": 37187.5,
        "min_reward": 0.0,
        "max_reward": 50000.0,
    },
    "bowling": {
        "env_id": "ALE/Bowling-v5",
        "solved_reward": 160.7,
        "min_reward": 0.0,
        "max_reward": 300.0,
    },
    "boxing": {
        "env_id": "ALE/Boxing-v5",
        "solved_reward": 12.1,
        "min_reward": -100.0,  # Can lose points
        "max_reward": 100.0,
    },
    "centipede": {
        "env_id": "ALE/Centipede-v5",
        "solved_reward": 12017.0,
        "min_reward": 0.0,
        "max_reward": 20000.0,
    },
    "chopper_command": {
        "env_id": "ALE/ChopperCommand-v5",
        "solved_reward": 7387.8,
        "min_reward": 0.0,
        "max_reward": 12000.0,
    },
    "crazy_climber": {
        "env_id": "ALE/CrazyClimber-v5",
        "solved_reward": 35829.4,
        "min_reward": 0.0,
        "max_reward": 50000.0,
    },
    "demon_attack": {
        "env_id": "ALE/DemonAttack-v5",
        "solved_reward": 1971.0,
        "min_reward": 0.0,
        "max_reward": 3000.0,
    },
    "double_dunk": {
        "env_id": "ALE/DoubleDunk-v5",
        "solved_reward": -15.5,
        "min_reward": -24.0,  # Basketball scoring system
        "max_reward": 24.0,
    },
    "fishing_derby": {
        "env_id": "ALE/FishingDerby-v5",
        "solved_reward": -38.7,
        "min_reward": -91.0,  # Fishing competition scoring
        "max_reward": 91.0,
    },
    "freeway": {
        "env_id": "ALE/Freeway-v5",
        "solved_reward": 29.6,
        "min_reward": 0.0,
        "max_reward": 50.0,
    },
    "frostbite": {
        "env_id": "ALE/Frostbite-v5",
        "solved_reward": 4534.4,
        "min_reward": 0.0,
        "max_reward": 8000.0,
    },
    "gopher": {
        "env_id": "ALE/Gopher-v5",
        "solved_reward": 2412.5,
        "min_reward": 0.0,
        "max_reward": 4000.0,
    },
    "gravitar": {
        "env_id": "ALE/Gravitar-v5",
        "solved_reward": 3351.4,
        "min_reward": 0.0,
        "max_reward": 6000.0,
    },
    "hero": {
        "env_id": "ALE/Hero-v5",
        "solved_reward": 30993.8,
        "min_reward": 0.0,
        "max_reward": 50000.0,
    },
    "ice_hockey": {
        "env_id": "ALE/IceHockey-v5",
        "solved_reward": 0.9,
        "min_reward": -10.0,  # Hockey scoring system
        "max_reward": 10.0,
    },
    "jamesbond": {
        "env_id": "ALE/Jamesbond-v5",
        "solved_reward": 302.8,
        "min_reward": 0.0,
        "max_reward": 500.0,
    },
    "kangaroo": {
        "env_id": "ALE/Kangaroo-v5",
        "solved_reward": 3035.0,
        "min_reward": 0.0,
        "max_reward": 5000.0,
    },
    "krull": {
        "env_id": "ALE/Krull-v5",
        "solved_reward": 2665.5,
        "min_reward": 0.0,
        "max_reward": 5000.0,
    },
    "kung_fu_master": {
        "env_id": "ALE/KungFuMaster-v5",
        "solved_reward": 22736.3,
        "min_reward": 0.0,
        "max_reward": 35000.0,
    },
    "montezuma_revenge": {
        "env_id": "ALE/MontezumaRevenge-v5",
        "solved_reward": 4753.3,
        "min_reward": 0.0,
        "max_reward": 8000.0,
    },
    "name_this_game": {
        "env_id": "ALE/NameThisGame-v5",
        "solved_reward": 8049.0,
        "min_reward": 0.0,
        "max_reward": 12000.0,
    },
    "pitfall": {
        "env_id": "ALE/Pitfall-v5",
        "solved_reward": 6463.7,
        "min_reward": 0.0,
        "max_reward": 10000.0,
    },
    "private_eye": {
        "env_id": "ALE/PrivateEye-v5",
        "solved_reward": 69571.3,
        "min_reward": 0.0,
        "max_reward": 100000.0,
    },
    "riverraid": {
        "env_id": "ALE/Riverraid-v5",
        "solved_reward": 17118.0,
        "min_reward": 0.0,
        "max_reward": 25000.0,
    },
    "road_runner": {
        "env_id": "ALE/RoadRunner-v5",
        "solved_reward": 7845.0,
        "min_reward": 0.0,
        "max_reward": 12000.0,
    },
    "robotank": {
        "env_id": "ALE/Robotank-v5",
        "solved_reward": 11.9,
        "min_reward": 0.0,
        "max_reward": 30.0,
    },
    "skiing": {
        "env_id": "ALE/Skiing-v5",
        "solved_reward": -12682.0,
        "min_reward": -20000.0,  # Penalty-based scoring
        "max_reward": 0.0,
    },
    "solaris": {
        "env_id": "ALE/Solaris-v5",
        "solved_reward": 12326.7,
        "min_reward": 0.0,
        "max_reward": 20000.0,
    },
    "star_gunner": {
        "env_id": "ALE/StarGunner-v5",
        "solved_reward": 10250.0,
        "min_reward": 0.0,
        "max_reward": 15000.0,
    },
    "tennis": {
        "env_id": "ALE/Tennis-v5",
        "solved_reward": -8.0,
        "min_reward": -24.0,  # Tennis scoring system
        "max_reward": 24.0,
    },
    "time_pilot": {
        "env_id": "ALE/TimePilot-v5",
        "solved_reward": 5229.2,
        "min_reward": 0.0,
        "max_reward": 8000.0,
    },
    "tutankham": {
        "env_id": "ALE/Tutankham-v5",
        "solved_reward": 167.6,
        "min_reward": 0.0,
        "max_reward": 300.0,
    },
    "up_n_down": {
        "env_id": "ALE/UpNDown-v5",
        "solved_reward": 11693.2,
        "min_reward": 0.0,
        "max_reward": 18000.0,
    },
    "venture": {
        "env_id": "ALE/Venture-v5",
        "solved_reward": 1187.5,
        "min_reward": 0.0,
        "max_reward": 2000.0,
    },
    "video_pinball": {
        "env_id": "ALE/VideoPinball-v5",
        "solved_reward": 17667.9,
        "min_reward": 0.0,
        "max_reward": 25000.0,
    },
    "wizard_of_wor": {
        "env_id": "ALE/WizardOfWor-v5",
        "solved_reward": 4756.5,
        "min_reward": 0.0,
        "max_reward": 8000.0,
    },
    "yars_revenge": {
        "env_id": "ALE/YarsRevenge-v5",
        "solved_reward": 54196.4,
        "min_reward": 0.0,
        "max_reward": 80000.0,
    },
    "zaxxon": {
        "env_id": "ALE/Zaxxon-v5",
        "solved_reward": 9173.3,
        "min_reward": 0.0,
        "max_reward": 15000.0,
    },
}
