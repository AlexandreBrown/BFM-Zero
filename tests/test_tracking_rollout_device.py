"""Rollout prompts must support CPU replay with a separate model device."""
import math
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.fb.agent import FBAgent
from humanoidverse.agents.fb.model import FBModelConfig, FBModelArchiConfig
from humanoidverse.agents.normalizers import ObsNormalizerConfig, BatchNormNormalizerConfig


class TrackingRolloutDeviceTests(unittest.TestCase):
    def test_nested_cpu_replay_is_moved_without_mutating_storage(self):
        observation = {"state": torch.randn(6, 4), "history": {"state": torch.randn(6, 2)}}

        def encode(obs):
            tree_map(lambda x: self.assertEqual(x.device.type, "meta"), obs)
            return torch.empty(6, 2, device="meta")

        agent = FBAgent.__new__(FBAgent)
        agent.cfg = NS(model=NS(seq_length=2))
        agent._model = NS(device="meta", backward_map=encode, project_z=lambda z: z)
        replay = NS(sample=Mock(return_value={"next": {"observation": observation}}))
        z = agent._sample_tracking_z({"expert_slicer": replay}, 2, 3)
        self.assertEqual(tuple(z.shape), (2, 3, 2))
        self.assertEqual(z.device.type, "meta")
        tree_map(lambda x: self.assertEqual(x.device.type, "cpu"), observation)
        replay.sample.assert_called_once_with(6, seq_length=3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA regression requires a GPU")
    def test_cpu_replay_with_real_cuda_encoder_and_normalizer(self):
        arch = FBModelArchiConfig(z_dim=8)
        arch = arch.model_copy(update={
            key: getattr(arch, key).model_copy(update={"hidden_dim": 32, "hidden_layers": 2})
            for key in ("f", "b", "actor")
        })
        cfg = FBModelConfig(
            device="cuda", archi=arch, seq_length=2,
            obs_normalizer=ObsNormalizerConfig(normalizers=BatchNormNormalizerConfig()),
        )
        model = cfg.build(gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32), 2)
        agent = FBAgent.__new__(FBAgent)
        agent.cfg, agent._model = NS(model=cfg), model
        observation = torch.randn(6, 4)
        replay = NS(sample=Mock(return_value={"next": {"observation": observation}}))
        mean = model._obs_normalizer._normalizers._normalizer.running_mean.clone()
        z = agent._sample_tracking_z({"expert_slicer": replay}, 2, 3)
        self.assertEqual(tuple(z.shape), (2, 3, 8))
        self.assertEqual(z.device.type, "cuda")
        self.assertTrue(torch.isfinite(z).all())
        torch.testing.assert_close(z.norm(dim=-1), torch.full((2, 3), math.sqrt(8), device="cuda"))
        torch.testing.assert_close(mean, model._obs_normalizer._normalizers._normalizer.running_mean)
        self.assertEqual(observation.device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA regression requires a GPU")
    def test_compiled_cuda_learning_update_from_cpu_replay(self):
        self._check_learning_update(device="cuda", compiled=True)

    def _check_learning_update(self, device, compiled):
        from humanoidverse.train import default_train_config
        from humanoidverse.agents.buffers.transition import DictBuffer

        cfg = default_train_config().agent
        arch = cfg.model.archi
        arch = arch.model_copy(update={
            **{key: getattr(arch, key).model_copy(update={"hidden_dim": 32, "hidden_layers": 2})
               for key in ("f", "b", "actor", "critic", "aux_critic", "discriminator")},
            "z_dim": 8,
        })
        cfg = cfg.model_copy(update={
            "model": cfg.model.model_copy(update={"archi": arch, "device": device, "seq_length": 2}),
            "train": cfg.train.model_copy(update={"batch_size": 8}),
            "compile": compiled, "cudagraphs": False,
        })
        spaces = gym.spaces.Dict({
            key: gym.spaces.Box(-np.inf, np.inf, (dim,), dtype=np.float32)
            for key, dim in (("state", 4), ("privileged_state", 6), ("last_action", 2), ("history_actor", 8))
        })
        agent = cfg.build(spaces, 2)
        obs = {key: torch.randn(16, *space.shape) for key, space in spaces.items()}
        next_obs = tree_map(torch.randn_like, obs)
        replay, expert = DictBuffer(capacity=16, device="cpu"), DictBuffer(capacity=16, device="cpu")
        replay.extend({
            "observation": obs, "action": torch.randn(16, 2),
            "next": {"observation": next_obs, "terminated": torch.zeros(16, 1, dtype=torch.bool)},
            "z": agent._model.sample_z(16),
            "aux_rewards": {key: torch.randn(16, 1) for key in cfg.aux_rewards},
        })
        expert.extend({"observation": obs, "next": {"observation": next_obs},
                       "done": torch.zeros(16, 1, dtype=torch.bool)})
        before = next(agent._model._actor.parameters()).detach().clone()
        for step in range(2):
            metrics = agent.update({"train": replay, "expert_slicer": expert}, step)
            for key, value in metrics.items():
                self.assertTrue(torch.isfinite(value).all(), key)
        self.assertFalse(torch.equal(before, next(agent._model._actor.parameters())))


if __name__ == "__main__":
    unittest.main()
