import unittest
import numpy as np
from envs.target import VTgtField

class VTgtFieldTest(unittest.TestCase):
    def setUp(self):
        self.version = 3
        self.dt = 0.01
        self.vtgt_field = VTgtField(version=self.version, dt=self.dt)
        self.vtgt_field.reset(version=self.version, seed=0)
        self.pose_agent = np.array([0.0, 0.0, 0.0])

    def test_update_field(self):
        t_sim = 1.0 
        x, y, th = 0.0, 0.0, 0.0

        for t in np.arange(0, t_sim, self.dt):
            pose = np.array([x, y, th])
            vtgt_local, flag_new_target = self.vtgt_field.update(pose)
            vtgt = self.vtgt_field.get_vtgt(pose[0:2])

            # Check that the velocity target has shape (2,)
            self.assertEqual(vtgt.shape, (2,))
            # Check that the values are finite
            self.assertTrue(np.all(np.isfinite(vtgt)))
            self.assertTrue(np.all(np.isfinite(vtgt_local[0])))
            self.assertTrue(np.all(np.isfinite(vtgt_local[1])))

            # Optionally check that velocity magnitude is non-negative
            R = np.sqrt(vtgt_local[0]**2 + vtgt_local[1]**2)
            self.assertTrue(np.all(R >= 0))

            # Integrate position
            x += vtgt[0].item() * self.dt
            y += vtgt[1].item() * self.dt

if __name__ == "__main__":
    unittest.main()
