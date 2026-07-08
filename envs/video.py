"""Video recording wrappers and training-video schedule resolution."""

from __future__ import annotations

import os

import gymnasium as gym

from envs.adapters import get_adapter


def wrap_eval_video(env: gym.Env, video_dir: str) -> gym.Env:
    """Record a full first episode during evaluation."""
    os.makedirs(video_dir, exist_ok=True)
    return gym.wrappers.RecordVideo(
        env,
        video_dir,
        episode_trigger=lambda episode_id: episode_id == 0,
        name_prefix="eval",
    )


def wrap_train_video(
    env: gym.Env,
    video_dir: str,
    *,
    num_envs: int,
    video_every_global_steps: int,
    video_length: int,
    name_prefix: str = "train",
) -> gym.Env:
    """Record fixed-length clips every *video_every_global_steps* global steps.

    Only applied to worker 0.  ``step_id`` is the cumulative env-step count
    for that worker; multiplying by ``num_envs`` approximates global steps.
    """
    os.makedirs(video_dir, exist_ok=True)

    def step_trigger(step_id: int) -> bool:
        # step_id==0 fires on the very first env step; skip to avoid calling
        # render() before headless MuJoCo (EGL/OSMesa) is fully ready.
        if step_id <= 0:
            return False
        return (step_id * max(1, num_envs)) % video_every_global_steps == 0

    return gym.wrappers.RecordVideo(
        env,
        video_dir,
        step_trigger=step_trigger,
        video_length=video_length if video_length > 0 else 1,
        name_prefix=name_prefix,
    )


def resolve_training_video_schedule(
    env_id: str,
    env_kwargs: dict | None,
    capture_video: bool,
    video_every_global_steps: int,
    total_timesteps: int,
    video_length_seconds: float,
) -> int:
    """Probe env metadata and return ``video_length_steps`` for training clips.

    Returns 0 when video recording is disabled, or when the env declares that
    it does not support training video (see :mod:`envs.adapters`).
    """
    if not capture_video or not video_every_global_steps or video_every_global_steps <= 0:
        return 0

    if not get_adapter(env_id).supports_training_video:
        print(
            f"Note: training RecordVideo clips are not supported for {env_id} "
            "(gym.make_vec). Disabling per-step video recording for this run."
        )
        return 0

    render_fps = None
    try:
        probe_env = gym.make(env_id, **(env_kwargs or {}))
        render_fps = getattr(probe_env, "metadata", {}).get("render_fps", None)
        probe_env.close()
    except Exception as e:
        print(f"Warning: could not probe env metadata for video settings: {e}")

    fps = int(render_fps) if render_fps else 30
    video_length_steps = max(1, int(round(float(video_length_seconds) * fps)))
    est_train_videos = max(1, int(total_timesteps // int(video_every_global_steps)))
    print(
        "Video estimate: "
        f"video_every_global_steps={video_every_global_steps}, "
        f"expected_train_videos≈{est_train_videos}, "
        f"video_length_seconds={video_length_seconds} "
        f"(render_fps={fps} => video_length_steps={video_length_steps})"
    )
    return video_length_steps
