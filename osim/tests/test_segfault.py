import os
import opensim
import unittest

from osim.env import L2RunEnv

opensim.Body('block', 0.0001 , opensim.Vec3(0), opensim.Inertia(1,1,.0001,0,0,0) );
opensim.Body('block', 0.0001 , opensim.Vec3(0), opensim.Inertia(1,1,.0001,0,0,0) );

model_path = os.path.join(os.path.dirname(__file__), '../models/gait9dof18musc.osim')

def _test(model_path, visualize):
    model = opensim.Model(model_path)
    brain = opensim.PrescribedController()
    model.addController(brain)
    state = model.initSystem()

    muscleSet = model.getMuscles()
    for j in range(muscleSet.getSize()):
        muscle = muscleSet.get(j)
        brain.addActuator(muscleSet.get(j))
        actuator_name = muscle.getName()
        func = opensim.Constant(1.0)
        brain.prescribeControlForActuator(actuator_name, func)

    block = opensim.Body('block', 0.0001 , opensim.Vec3(0), opensim.Inertia(1,1,.0001,0,0,0) );
    model.addComponent(block)
    pj = opensim.PlanarJoint('pin',
                            model.getGround(), # PhysicalFrame
                            opensim.Vec3(0, 0, 0),
                            opensim.Vec3(0, 0, 0),
                            block, # PhysicalFrame
                            opensim.Vec3(0, 0, 0),
                            opensim.Vec3(0, 0, 0))
    model.addComponent(pj)
    model.initSystem()
    pj.getCoordinate(1)

class SegfaultTest(unittest.TestCase):
    def test_seg_fault(self):
        _test(model_path, False)
        _test(model_path, False)

        env = L2RunEnv(visualize=False)
        env1 = L2RunEnv(visualize=False)

        env1.reset()
        r1 = env1.get_reward()
        self.assertGreaterEqual(r1, 0, "Reward should be positive")

        env.reset()
        r2 = env.get_reward()
        self.assertGreaterEqual(r2, 0, "Reward should be positive")

if __name__ == '__main__':
    unittest.main()