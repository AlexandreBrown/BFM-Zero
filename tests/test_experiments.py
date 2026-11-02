"""Operational contracts of the experiment layer; no simulator or network."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np

from experiments import train as launch
from humanoidverse import train


class ExperimentTests(unittest.TestCase):
    def args(self, *options):
        return launch.parser().parse_args(["--work-dir", "/tmp/bfm-test", *options])

    def test_presets_validate_without_booting_a_simulator(self):
        with patch.object(train.TrainConfig, "build") as build:
            for name in ("smoke", "pilot", "reference"):
                cfg = launch.resolve_config(self.args("--preset", name))
                self.assertEqual(cfg.agent.name, "FBcprAuxAgent")
                self.assertEqual(cfg.wandb_ename, "glen-berseth")
            build.assert_not_called()
        reference = launch.resolve_config(self.args("--preset", "reference"))
        native = train.default_train_config()
        self.assertEqual(reference.agent, native.agent)
        self.assertEqual(reference.env, native.env)
        self.assertEqual(reference.buffer_size, native.buffer_size)

    def test_invalid_overrides_do_not_silently_launch_a_different_experiment(self):
        for expression in ("agent.train.lr_actorr=0.1", "missing.x=1", "no_equals"):
            with self.assertRaises(ValueError):
                launch.resolve_config(self.args("--set", expression))
        for expression in ("agent.train.batch_size=65", "online_parallel_envs=33", "buffer_size=32",
                           "agent.train.expert_asm_ratio=0.99"):
            with self.assertRaises(ValueError):
                launch.resolve_config(self.args("--set", expression))

    def test_credentials_rejected_before_config_can_be_written(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            path.write_text(json.dumps({"tags": {"api_key": "test-placeholder"}}))
            with self.assertRaisesRegex(ValueError, "Credentials"):
                launch.resolve_config(self.args("--config", str(path)))

    def test_resume_uses_saved_architecture_and_rejects_changed_learning_rate(self):
        cfg = launch.resolve_config(self.args("--seed", "7"))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "checkpoint").mkdir()
            (root / "config.json").write_text(cfg.model_dump_json())
            (root / "checkpoint/config.json").write_text(cfg.agent.model_dump_json())
            args = self.args("--resume", "--work-dir", folder, "--set", "num_env_steps=8192")
            resumed = launch.resolve_config(args)
            self.assertEqual(resumed.seed, 7)
            self.assertEqual(resumed.agent, cfg.agent)
            train.validate_resume_config(root, resumed)
            args.overrides.append("agent.train.lr_actor=0.1")
            with self.assertRaisesRegex(ValueError, "agent"):
                train.validate_resume_config(root, launch.resolve_config(args))

    def test_hydra_replaces_a_key_instead_of_creating_duplicate_overrides(self):
        cfg = launch.resolve_config(self.args("--hydra", "robot.control.action_scale=0.1"))
        self.assertEqual([v for v in cfg.env.hydra_overrides if v.startswith("robot.control.action_scale=")],
                         ["robot.control.action_scale=0.1"])

    def test_lfs_pointer_is_rejected_without_unpickling(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "motion.pkl"
            path.write_text("version https://git-lfs.github.com/spec/v1\n")
            with self.assertRaisesRegex(ValueError, "LFS pointer"):
                launch.check_data(path)

    def test_native_wandb_keeps_logs_in_run_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = train.default_train_config().model_copy(update={"work_dir": folder})
            with patch.object(train.wandb, "init") as init, patch.dict("os.environ", {"WANDB_NAME": "smoke-test"}):
                train.init_wandb(cfg)
            self.assertEqual(init.call_args.kwargs["dir"], str(Path(folder) / "_wandb"))
            self.assertEqual(init.call_args.kwargs["name"], "smoke-test")

    def test_training_failure_is_reported_to_wandb_and_reraised(self):
        import torch
        import wandb
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                data = root / "motion.pkl"
                data.write_bytes(b"test-motion-payload")
                args = self.args("--work-dir", str(root / "run"), "--data-path", str(data))
                cfg = launch.resolve_config(args)
                workspace = SimpleNamespace(train=Mock(side_effect=RuntimeError("compiler failed") if failure else None))
                with patch.dict(os.environ), \
                     patch.object(torch.cuda, "is_available", return_value=True), \
                     patch.object(torch.cuda, "get_device_name", return_value="test-gpu"), \
                     patch("experiments.preflight.check_compile_toolchain"), \
                     patch.object(train.TrainConfig, "build", return_value=workspace), \
                     patch.object(wandb, "finish") as finish, contextlib.redirect_stdout(io.StringIO()):
                    if failure:
                        with self.assertRaisesRegex(RuntimeError, "compiler failed"):
                            launch.run(args, cfg)
                    else:
                        launch.run(args, cfg)
                    finish.assert_called_once_with(exit_code=int(failure))

    def test_cli_failure_preserves_exit_status_despite_simulator_shutdown_hook(self):
        code = """
import atexit, os
from experiments import train
atexit.register(lambda: os._exit(0))
def fail():
    raise RuntimeError('test compilation failure')
train.main = fail
train.cli()
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=launch.ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("test compilation failure", result.stderr)

    def test_evaluation_preserves_protocol_and_writes_native_numeric_results(self):
        from experiments.evaluate import main
        from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
        from humanoidverse.agents.evaluations.humanoidverse_isaac import HumanoidVerseIsaacTrackingEvaluationConfig
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "motion.pkl"
            data.write_bytes(b"test-payload-not-unpickled")
            (root / "config.json").write_text(train.default_train_config().model_dump_json())
            env = SimpleNamespace(close=Mock())
            model = SimpleNamespace(eval=Mock())
            result = {"motion": {"emd": np.float32(0.5)}}
            evaluation = SimpleNamespace(run=Mock(return_value=(result, {"emd": np.float32(0.5)})))
            with patch("humanoidverse.agents.load_utils.load_model_from_checkpoint_dir", return_value=model), \
                 patch.object(HumanoidVerseIsaacConfig, "build", return_value=(env, {})) as build, \
                 patch.object(HumanoidVerseIsaacTrackingEvaluationConfig, "build", return_value=evaluation), \
                 contextlib.redirect_stdout(io.StringIO()):
                main(["--model-folder", str(root), "--data-path", str(data), "--seed", "1"])
            report = json.loads(next((root / "evaluations").glob("*.json")).read_text())
            self.assertEqual(report["lookahead_frames"], 1)
            self.assertEqual(report["aggregate"]["emd"], 0.5)
            self.assertEqual(report["seed"], 1)
            build.assert_called_once_with(32)
            env.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
