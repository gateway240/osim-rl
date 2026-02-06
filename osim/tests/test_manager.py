import unittest
import opensim

class OpenSimIntegrationTest(unittest.TestCase):
    def test_muscle_controller_integration(self):
        stepsize = 0.01

        # Load existing model
        model_path = "osim/models/gait9dof18musc.osim"
        model = opensim.Model(model_path)
        model.setUseVisualizer(False)

        # Build the controller
        brain = opensim.PrescribedController()
        controllers = []

        state = model.initSystem()
        muscleSet = model.getMuscles()
        forceSet = model.getForceSet()

        for j in range(muscleSet.getSize()):
            muscle = muscleSet.get(j)
            func = opensim.Constant(1.0)
            controllers.append(func)
            brain.addActuator(muscle)
            actuator_name = muscle.getName()
            brain.prescribeControlForActuator(actuator_name, func)

        model.addController(brain)

        # Add a block with PinJoint
        blockos = opensim.Body('blockos', 0.0001, opensim.Vec3(0), opensim.Inertia(1,1,.0001,0,0,0))
        pj = opensim.PinJoint(
            "pinblock",
            model.getGround(),
            opensim.Vec3(-0.5,0,0),
            opensim.Vec3(0,0,0),
            blockos,
            opensim.Vec3(0,0,0),
            opensim.Vec3(0,0,0)
        )
        bodyGeometry = opensim.Ellipsoid(0.1, 0.1, 0.1)
        bodyGeometry.setColor(opensim.Gray)
        blockos.attachGeometry(bodyGeometry)

        model.addComponent(pj)
        model.addComponent(blockos)

        # Contact geometry
        block = opensim.ContactSphere(0.4, opensim.Vec3(0,0,0), blockos)
        model.addContactGeometry(block)

        # Reinitialize the system
        state0 = model.initSystem()
        state = opensim.State(state0)

        # Modify muscle
        muscleSet.get(0).setMaxIsometricForce(100000.0)

        # Get ligaments
        ligamentSet = []
        for j in range(20, 26):
            ligamentSet.append(opensim.CoordinateLimitForce.safeDownCast(forceSet.get(j)))

        state.setTime(0)
        manager = opensim.Manager(model)
        manager.setIntegratorAccuracy(5e-4)
        manager.initialize(state)

        # Run a few integration steps
        for i in range(5):  # reduced from 20 for faster test
            # Set excitation values
            for j in range(muscleSet.getSize()):
                controllers[j].setValue(((i + j) % 10) * 0.1)

            # Integrate
            t = (i + 1) * stepsize
            manager.integrate(t)

            # Realize dynamics
            model.realizeDynamics(state)

            # Assert activations and excitations are in [0,1]
            activation = muscleSet.get(0).getActivation(state)
            excitation = muscleSet.get(0).getExcitation(state)
            self.assertGreaterEqual(activation, 0)
            self.assertLessEqual(activation, 1)
            self.assertGreaterEqual(excitation, 0)
            self.assertLessEqual(excitation, 1)

            # Check ligaments produce a numeric force
            for lig in ligamentSet:
                force = lig.calcLimitForce(state)
                self.assertIsInstance(force, float)

        # Optional: test resetting the state
        state = opensim.State(state0)
        self.assertEqual(state.getTime(), state0.getTime())


if __name__ == "__main__":
    unittest.main()
