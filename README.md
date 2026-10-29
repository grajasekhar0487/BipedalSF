# BipedalSF

An installable downstream Isaac Lab task package with a standard uv `src` layout.

The registered task is:

- `Bipedalsf-Locomotion-SF-Tron2A-v0` — flat-ground standing balance and commanded biped walking with RSL-RL PPO.


## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the project environment. The default
environment uses the Newton backend and does not install Isaac Sim:

```bash
uv sync
```

This project's `pyproject.toml` uses editable relative paths to an Isaac Lab source checkout. Update
`[tool.uv.sources]` if either directory moves.

The project forwards every optional extra declared by the Isaac Lab package used to create it, including physics,
rendering, visualizer, RL, teleoperation, and development features. Inspect `pyproject.toml` for the complete list and
pass each required extra to `uv run`:

```bash
# Standalone OV PhysX
uv run --extra ovphysx isaaclab random_agent --task <TASK_NAME> physics=ovphysx

# Isaac Sim with PhysX and RTX rendering
uv run --extra isaacsim isaaclab random_agent --task <TASK_NAME> physics=isaacsim_physx

# Isaac Sim with the Rerun visualizer
uv run --extra isaacsim --extra rerun isaaclab random_agent --task <TASK_NAME> --viz rerun
```

The `all` extra forwards Isaac Lab's `all` selection. The `ov` extra installs both the `ovphysx` and `ovrtx` runtimes;
you can also select either independently. Commit `pyproject.toml` and `uv.lock` so collaborators use the same
environment.

## Run the tasks

The environment uses the supplied `SF_TRON2A/usd/robot.usd` articulation and offers `isaacsim_physx` and
`newton_mjwarp` physics presets. Training defaults to the Newton preset and RSL-RL. Select `--device cpu` or a
CUDA device such as `--device cuda:0`; Newton MJWarp is intended for GPU training, while CPU training is available
with the Isaac Sim PhysX preset.

For Isaac Sim rendering, install the `isaacsim` extra and use `--viz kit`. For Newton rendering, use
`--viz newton_gl`; that opens Newton's visualizer while training with MJWarp.

```bash
# List this project's environments and available physics presets
uv run isaaclab list_envs --show_presets

# Train with Newton MJWarp and render in Newton GL
uv run isaaclab train --rl_library rsl_rl --task Bipedalsf-Locomotion-SF-Tron2A-v0 \
  --num_envs 256 --device cuda:0 --viz newton_gl physics=newton_mjwarp

# Train with Isaac Sim PhysX and render in its Kit viewport
uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task Bipedalsf-Locomotion-SF-Tron2A-v0 \
  --num_envs 256 --device cuda:0 --viz kit physics=isaacsim_physx

# Train with Isaac Sim PhysX on CPU (reduce the environment count for CPU throughput)
uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task Bipedalsf-Locomotion-SF-Tron2A-v0 \
  --num_envs 16 --device cpu --viz kit physics=isaacsim_physx

# Play a trained policy with rendering
uv run --extra isaacsim isaaclab play --rl_library rsl_rl --task Bipedalsf-Locomotion-SF-Tron2A-v0 \
  --checkpoint latest --num_envs 1 --device cuda:0 --viz kit physics=isaacsim_physx

# Distributed training
uv run isaaclab train_multigpu --rl_library rsl_rl --task Bipedalsf-Locomotion-SF-Tron2A-v0 --num_gpus 2

# Benchmark startup, runtime, training, or play
uv run isaaclab benchmark runtime --task Bipedalsf-Locomotion-SF-Tron2A-v0 --num_envs 16 --num_steps 1000
uv run isaaclab benchmark training --rl_library rsl_rl --task Bipedalsf-Locomotion-SF-Tron2A-v0 --max_iterations 10
```

Use `physics=<PRESET>` to select one of the presets shown by `isaaclab list_envs`.

## Project structure

Tasks live in `src/BipedalSF/tasks`. The SF-TRON2A environment, task registration, and RSL-RL agent configuration live
under `tasks/locomotion/config/sf_tron2a`; its shared robot articulation configuration is in `assets/robots/sf_tron2a.py`.

## Project assets

Keep project-owned USD files and related data under `src/BipedalSF/assets/data`. The asset module exposes a
stable path that works from an editable checkout and an installed wheel:

```python
from BipedalSF.assets import BIPEDALSF_ASSETS_DIR

robot_usd_path = BIPEDALSF_ASSETS_DIR / "robots" / "SF_TRON2A" / "usd" / "robot.usd"
```

Keep asset configuration in the `BipedalSF.assets` package and pass `str(robot_usd_path)` to configuration fields that
expect a string. USD files are configured for Git LFS by the project's `.gitattributes`; install Git LFS before adding
large binary assets.

## Development

Run the registration test and code-quality checks through the project environment:

```bash
uv run pytest
uv run pre-commit run --all-files
```

The test helpers under `source/isaaclab_tasks/test` in the Isaac Lab repository are not part of the installed
`isaaclab_tasks` package. Keep test fixtures in this project and use public Isaac Lab APIs. If you copy
`env_test_utils.py`, it becomes vendored code whose upstream changes you must track.

To configure VS Code or Cursor, run the `setup_python_env` task or invoke its command directly:

```bash
uv run isaaclab --editor
```

The setup command selects the active interpreter and writes a git-ignored `pyrightconfig.json`. The resulting
configuration inherits the project's checked-in Pyright settings and adds the Isaac Sim extensions, project `src` root,
and any Isaac Lab packages discovered in the active Python environment. This supports both Pylance in VS Code and
basedpyright in Cursor.

In VS Code, use Pylance and select the interpreter that ran the setup command. In Cursor, install the
[basedpyright extension](https://marketplace.visualstudio.com/items?itemName=detachhead.basedpyright) instead of
Pylance, select the same interpreter, and reload the window. Both language servers read `pyrightconfig.json`.

When using the `isaacsim` extra, include it while generating the editor configuration so the command can discover the
Isaac Sim installation:

```bash
uv run --extra isaacsim isaaclab --editor
```

For an Isaac Sim binaries installation that is not available in the project environment, provide its path explicitly:

```bash
# Linux
uv run isaaclab --editor --isaac_path <isaac-sim-path>

# Windows
uv run isaaclab --editor --isaac_path <isaac-sim-path>
```

## Troubleshooting

If Pylance or basedpyright cannot resolve modules, confirm that the selected interpreter matches the one used to run the
setup command, then reload the editor window. To add a missing extension or reduce indexing memory, edit the `extraPaths`
array in the root `pyrightconfig.json`; remove simulator extension directories that the project does not use.
