#include <boost/bind.hpp>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <stdio.h>
#include <ros/console.h>
#include <gazebo/common/Exception.hh>
#include <map>

#include <ros/ros.h>

#include <autopilot_simulator/QuadroControlData.h>

namespace gazebo
{
    class ModelPush : public ModelPlugin
    {
        public: 

        void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
        {
            // Store the pointer to the model
            // node.reset(new transport::Node());
            // node->Init(_parent->GetName());
            // pub = node->Advertise<msgs::WorldControl>("~/world_control");

            client = nh.serviceClient<autopilot_simulator::QuadroControlData>("process_quadro_control");


            model   = _parent;
            link    = model->GetLink();
            world = model->GetWorld();
            physics::PhysicsEnginePtr eng = world->GetPhysicsEngine();

            eng->SetMaxStepSize(0.001);

            // Listen to the update event. This event is broadcast every
            // simulation iteration.
            updateConnection = event::Events::ConnectWorldUpdateBegin(
                boost::bind(&ModelPush::OnUpdate, this, _1));

            ROS_INFO_STREAM("Sensors: " << model->GetSensorCount());

            cntc_mgr = eng->GetContactManager();
            
            ROS_INFO_STREAM("Sensors init: " << world->SensorsInitialized());



            // GAZEBO_SENSORS_USING_DYNAMIC_POINTER_CAST;
            // sensors::ContactSensorPtr child = dynamic_pointer_cast<sensors::ContactSensor>( world->GetEntity("contact_sensor") );
            // if ( child == NULL )
            // {
            //     ROS_INFO_STREAM("No child with name 'contact_sensor'");
            // }

            // world->SetPaused(true);
            // world->Reset();

            // world->SetPaused(false);
            // world->PrintEntityTree();

            // initial_iter = world->GetIterations();
            // world->RunBlocking(25);

            std::cout << "Publishing Load." << std::endl;
            prev_upd_time = world->GetSimTime();
        }

        uint32_t initial_iter = 0;
        double torque_x = 0.0001;
        int skip = 0;

        #define SIGN(x) ((x > 0) ? 1 : ((x < 0) ? -1 : 0))

        common::Time prev_upd_time;

        // Called by the world update start event
        void OnUpdate(const common::UpdateInfo & /*_info*/)
        {
            // std::cout << "Received OnUpdate." << std::endl;
            // ROS_INFO_STREAM("Wall time: " << world->GetWallTime());
            common::Time current_time = world->GetSimTime();

            if ( current_time - prev_upd_time >= common::Time(0, 5000000) )
            {
                prev_upd_time = current_time;

                math::Pose      pose        = model->GetWorldPose();
                math::Vector3   rot         = pose.rot.GetAsEuler();
                math::Vector3   linear_vel  = link->GetRelativeLinearVel();

                ROS_INFO_STREAM("Iter: " << world->GetIterations() << " Time: " << current_time);
                // ROS_INFO_STREAM("ex: " << pose.pos.x << " ey: " << pose.pos.y);

                // Apply a small linear velocity to the model.
                // model->SetLinearVel(math::Vector3(.03, 0, 0));

                autopilot_simulator::QuadroControlData srv_req;
                srv_req.request.ex = pose.pos.x;
                srv_req.request.ey = pose.pos.y;

                if ( client.call( srv_req ) )
                {
                    // ROS_INFO("Force: %f %f", srv_req.response.x_force, srv_req.response.y_force);
                }
                else
                {
                    ROS_ERROR("Failed to call service");
                    return;
                }

                
                // printf("%f / %f / %f / %f / %f\n", rot.x, rot.y, rot.z, rot.x * 180 / M_PI * SIGN(torque_x), torque_x);
                // printf("%f / %f / %f\n", linear_vel.x, linear_vel.y, linear_vel.z);
                
                
                // common::Time::MSleep(2000);


                // usleep( 1 * 1000 );

                applied_force = math::Vector3(1, 0, 0);
                
                /*
                if ( rot.x * 180 / M_PI * SIGN(torque_x) > 10 )
                    torque_x *= -1;
                
                link->AddRelativeTorque(math::Vector3(torque_x, 0, 0));
                */

                if ( cntc_mgr->GetContactCount() ) 
                    ROS_INFO_STREAM("Contacts: " << cntc_mgr->GetContactCount());
                
                

                // if ( skip < 25 )
                // {
                //     skip++;
                //     world->Step(1);
                // } else {
                //     world->SetPaused(true);
                // }

            }

            link->AddRelativeForce(applied_force);  // [Newtons]

            // math::Vector3 force = link->GetRelativeForce();
            // printf("Forces: %f / %f / %f\n", force.x, force.y, force.z);


            // std::cout << "Publishing OnUpdate." << std::endl;
        }

        private:
            // transport::PublisherPtr pub;
            // transport::NodePtr node;

            math::Vector3 applied_force;
            ros::ServiceClient client;
            ros::NodeHandle nh;


            physics::ContactManager *cntc_mgr;
            physics::WorldPtr world;
            physics::LinkPtr link;
            physics::ModelPtr model;
            event::ConnectionPtr updateConnection;
    };

    // Register this plugin with the simulator
    GZ_REGISTER_MODEL_PLUGIN(ModelPush)
}