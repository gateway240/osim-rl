import unittest
from osim.env import L2M2019Env

class TestL2M2019EnvRollback(unittest.TestCase):
    def test_state_rollback(self):
        env = L2M2019Env(visualize=False)
        env.reset()

        state_old = None

        for s in range(80):
            if s == 30:
                state_old = env.osim_model.get_state()
                self.assertIsNotNone(state_old)

            if s % 50 == 49:
                env.osim_model.set_state(state_old)
                self.assertAlmostEqual(env.osim_model.get_state().getTime(),
                                       state_old.getTime())

            o, r, d, i = env.step(env.action_space.sample())

            self.assertIsInstance(o, (list, dict))
            self.assertIsInstance(r, (int, float))
            self.assertIsInstance(d, bool)

if __name__ == "__main__":
    unittest.main()
