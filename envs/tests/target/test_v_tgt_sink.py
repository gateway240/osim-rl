import unittest
import numpy as np
from envs.target.v_tgt_field import VTgtSink

class VTgtSinkTest(unittest.TestCase):
    def setUp(self):
        rng_xy = np.array([[-20, 20], [-20, 20]])
        self.vtgt_obj = VTgtSink(rng_xy, res_map=np.array([1, 1]), res_get=np.array([3, 1]))
        self.p_sink = np.array([13.8, 2.5])
        self.d_sink = np.linalg.norm(self.p_sink)
        v_amp_rng = np.array([1.0, 2.0])
        self.vtgt_obj.create_vtgt_sink(self.p_sink, self.d_sink, v_amp_rng, v_phase0=np.pi)

    def test_get_vtgt_at_origin(self):
        pose = np.array([0.0, 0.0, 0.0])
        vtgt = self.vtgt_obj.get_vtgt(pose[0:2])
        self.assertEqual(vtgt.shape, (2,))
        self.assertTrue(np.all(np.isfinite(vtgt)))

    def test_vtgt_field_local(self):
        # Sweep along a path towards the sink
        for x, y, th in zip(np.linspace(0, self.p_sink[0], 30),
                            np.linspace(0, self.p_sink[1], 30),
                            np.linspace(0, 90*np.pi/180, 30)):
            pose = np.array([x, y, th])
            vtgt_local = self.vtgt_obj.get_vtgt_field_local(pose)
            # Check shapes
            self.assertEqual(vtgt_local[0].shape, self.vtgt_obj._generate_grid(self.vtgt_obj.rng_get, self.vtgt_obj.res_get)[0].shape)
            self.assertEqual(vtgt_local[1].shape, vtgt_local[0].shape)
            # Check that values are finite
            self.assertTrue(np.all(np.isfinite(vtgt_local[0])))
            self.assertTrue(np.all(np.isfinite(vtgt_local[1])))

    def test_vtgt_field_magnitude(self):
        # Velocity magnitude should be non-negative
        pose = np.array([0.0, 0.0, 0.0])
        U, V = self.vtgt_obj.get_vtgt_field_local(pose)
        R = np.sqrt(U**2 + V**2)
        self.assertTrue(np.all(R >= 0))

if __name__ == "__main__":
    unittest.main()
