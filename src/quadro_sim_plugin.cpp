#include <boost/bind.hpp>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <stdio.h>

namespace gazebo
{
    class ModelPush : public ModelPlugin
    {
        public: 

        void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
        {
            // Store the pointer to the model
            model   = _parent;
            link    = model->GetLink();
            physics::WorldPtr world = model->GetWorld();
            physics::PhysicsEnginePtr eng = world->GetPhysicsEngine();

            eng->SetMaxStepSize(0.001);

            // Listen to the update event. This event is broadcast every
            // simulation iteration.
            updateConnection = event::Events::ConnectWorldUpdateBegin(
                boost::bind(&ModelPush::OnUpdate, this, _1));


        }

        double torque_x = 0.0001;

        #define SIGN(x) ((x > 0) ? 1 : ((x < 0) ? -1 : 0))

        // Called by the world update start event
        void OnUpdate(const common::UpdateInfo & /*_info*/)
        {
            // Apply a small linear velocity to the model.
            // model->SetLinearVel(math::Vector3(.03, 0, 0));
            
            math::Pose      pose        = model->GetWorldPose();
            math::Vector3   rot         = pose.rot.GetAsEuler();
            math::Vector3   linear_vel  = link->GetRelativeLinearVel();
            // printf("%f / %f / %f\n", pose.pos.x, pose.pos.y, pose.pos.z);
            // printf("%f / %f / %f / %f / %f\n", rot.x, rot.y, rot.z, rot.x * 180 / M_PI * SIGN(torque_x), torque_x);
            printf("%f / %f / %f\n", linear_vel.x, linear_vel.y, linear_vel.z);
            
            math::Vector3 force = link->GetRelativeForce();
            // printf("%f / %f / %f\n", force.x, force.y, force.z);

            // usleep( 1 * 1000 );

            
            /*
            if ( rot.x * 180 / M_PI * SIGN(torque_x) > 10 )
                torque_x *= -1;
            
            link->AddRelativeTorque(math::Vector3(torque_x, 0, 0));
            */

            link->AddRelativeForce(math::Vector3(0.01, 0, 0));  // [Newtons]
        }

        private:
            physics::LinkPtr link;
            physics::ModelPtr model;
            event::ConnectionPtr updateConnection;
    };

    // Register this plugin with the simulator
    GZ_REGISTER_MODEL_PLUGIN(ModelPush)
}