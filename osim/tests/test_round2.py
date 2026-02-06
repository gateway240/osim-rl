import unittest
import numpy as np
import math
from osim.env import ProstheticsEnv, rect

class SimulationTest(unittest.TestCase):
    def test_reset_types(self):
        env = ProstheticsEnv(visualize=False, difficulty=0)
        o = env.reset()
        self.assertIsInstance(o, dict)
        o = env.reset(project=False)
        self.assertIsInstance(o, dict)
        o = env.reset(project=True)
        self.assertIsInstance(o, dict)

    def test_step_project_false(self):
        env = ProstheticsEnv(visualize=False, difficulty=0)
        env.reset()
        action = env.action_space.sample()
        o,r,d,i = env.step(action, project=False)
        with self.assertRaises(KeyError):
            _ = o["target_vel"]

        env = ProstheticsEnv(visualize=False, difficulty=1)
        env.reset()
        o,r,d,i = env.step(action, project=False)
        self.assertEqual(len(o["target_vel"]), 3)

    def test_targets_and_speed(self):
        env = ProstheticsEnv(visualize=False, difficulty=1)
        env.reset()
        env.generate_new_targets(10)
        action = env.action_space.sample()
        for _ in range(20):
            o,r,d,i = env.step(action, project=False)

        self.assertGreater(rect([2, 0])[0], 1.99)
        self.assertLess(rect([2, math.pi/2.0])[0], 0.01)

        env.reset()
        env.generate_new_targets(10)
        self.assertGreater(np.sum((env.targets[300,:] - np.array([1.25,0,0]))**2), 0.01)

        # Test velocity effect
        state = env.osim_model.get_state()
        env.osim_model.get_joint("ground_pelvis").get_coordinates(0).setSpeedValue(state, 5)
        env.osim_model.set_state(state)
        o1,r1,d,i = env.step(action, project=False)

        state = env.osim_model.get_state()
        env.osim_model.get_joint("ground_pelvis").get_coordinates(0).setSpeedValue(state, 1.25)
        env.osim_model.set_state(state)
        o2,r2,d,i = env.step(action, project=False)

        self.assertGreater(o1["joint_vel"]["ground_pelvis"], o2["joint_vel"]["ground_pelvis"])
        self.assertGreater(r2,r1)

if __name__ == '__main__':
    unittest.main()
