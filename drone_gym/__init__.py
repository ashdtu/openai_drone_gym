from gym.envs.registration import register

register(
    id='airsim-drone-v0',
    entry_point='drone_gym.envs:DroneAirsim',
)

