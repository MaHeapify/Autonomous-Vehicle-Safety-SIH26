from metadrive import MetaDriveEnv

config = {
    "use_render": True,
    "start_seed": 0,
    "map": "maps/opendrive/yelahanka.xodr",
    "num_agents": 1,
}

env = MetaDriveEnv(config)

try:
    obs, info = env.reset()

    print("Yelahanka OpenDRIVE loaded successfully!")
    print("Observation received.")

    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

finally:
    env.close()