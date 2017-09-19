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

            // Listen to the update event. This event is broadcast every
            // simulation iteration.
            updateConnection = event::Events::ConnectWorldUpdateBegin(
                boost::bind(&ModelPush::OnUpdate, this, _1));


        }

        // Called by the world update start event
        void OnUpdate(const common::UpdateInfo & /*_info*/)
        {
            // Apply a small linear velocity to the model.
            // model->SetLinearVel(math::Vector3(.03, 0, 0));
            
            math::Vector3 linear_vel = link->GetRelativeLinearVel();
            printf("%f / %f / %f\n", linear_vel.x, linear_vel.y, linear_vel.z);

            math::Vector3 force = link->GetRelativeForce();
            printf("%f / %f / %f\n", force.x, force.y, force.z);

            usleep( 1 * 1000 );

            link->AddRelativeForce(math::Vector3(0, 0, 0.02));
        }

        private:
            physics::LinkPtr link;
            physics::ModelPtr model;
            event::ConnectionPtr updateConnection;
    };

    // Register this plugin with the simulator
    GZ_REGISTER_MODEL_PLUGIN(ModelPush)
}