import unittest
import opensim

class OpenSimBallTest(unittest.TestCase):
    def test_ball_free_joint_simulation(self):
        stepsize = 0.01

        # Load model
        model_path = "osim/models/gait9dof18musc.osim"
        model = opensim.Model(model_path)
        model.setUseVisualizer(False)

        # Create a tiny ball
        r = 1e-6
        ballBody = opensim.Body(
            'ball',
            0.0001,
            opensim.Vec3(0),
            opensim.Inertia(1,1,0.0001,0,0,0)
        )
        ballGeometry = opensim.Ellipsoid(r, r, r)
        ballGeometry.setColor(opensim.Gray)
        ballBody.attachGeometry(ballGeometry)

        # Attach ball to model via FreeJoint
        ballJoint = opensim.FreeJoint(
            "weldball",
            model.getGround(),
            opensim.Vec3(0,0,0),
            opensim.Vec3(0,0,0),
            ballBody,
            opensim.Vec3(0,0,0),
            opensim.Vec3(0,0,0)
        )
        model.addComponent(ballJoint)
        model.addComponent(ballBody)

        # Add contact geometry
        ballContact = opensim.ContactSphere(r, opensim.Vec3(0,0,0), ballBody)
        model.addContactGeometry(ballContact)

        # Initialize system
        state = model.initSystem()
        for i in range(6):
            ballJoint.getCoordinate(i).setLocked(state, True)
        state.setTime(0)

        manager = opensim.Manager(model)
        manager.setIntegratorAccuracy(5e-4)
        manager.initialize(state)

        t = 0.0
        for i in range(10):  # reduced steps for fast test
            t_next = t + stepsize
            manager.integrate(t_next)
            t = t_next

            # Restart simulation every 5 frames
            if (i + 1) % 5 == 0:
                newloc = opensim.Vec3(float(i) / 5, 0, 0)
                opensim.PhysicalOffsetFrame.safeDownCast(ballJoint.getChildFrame()).set_translation(newloc)

                r_new = i * 0.005
                ballContact.setRadius(r_new)

                # Reinitialize state and manager
                state = model.initializeState()
                ballJoint.getCoordinate(3).setValue(state, i / 100.0)
                for j in range(6):
                    ballJoint.getCoordinate(j).setLocked(state, True)
                manager.initialize(state)
                t = state.getTime()

                # Assertions: ball radius and joint coords
                self.assertAlmostEqual(ballContact.getRadius(), r_new)
                self.assertAlmostEqual(ballJoint.getCoordinate(3).getValue(state), i / 100.0)

        # Final checks: time should be advancing
