from gymnasium.envs.registration import register
register(
    id="HalfCheetahVel-v1",
    entry_point="envs.custom_envs.halfcheetahvel:make_target_vel_env",
    kwargs={"target_vel": 1.0, "render_mode":"rgb_array"}
)
register(
    id="HalfCheetahVelGoal-v1",
    entry_point="envs.custom_envs.halfcheetahvel_goal:make_halfcheetahvel_goal_env",
    kwargs={
        "schedule_mode": "fixed",
        "velocities": [1.0],
        "switch_after_steps": [],
        "render_mode": "rgb_array",
    },
)
register(
    id="AntDir-v1",
    entry_point="envs.custom_envs.antdir:make_target_dir_env",
    kwargs={"target_dir": [0.0, 1.0], "render_mode":"rgb_array"}
)
register(
    id="AntDirGoal-v1",
    entry_point="envs.custom_envs.antdir_goal:make_antdir_goal_env",
    kwargs={
        "schedule_mode": "fixed",
        "directions": [[1.0, 0.0], [-1.0, 0.0]],
        "switch_after_steps": [500],
        "render_mode": "rgb_array",
    },
)
