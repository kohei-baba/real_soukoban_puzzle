// Force-limited mobile pusher + synchronous Gazebo stepping.
// The crate is NEVER commanded during a STEP. Only RESET teleports bodies.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <sdf/Element.hh>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/msgs/world_control.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/transport/Node.hh>
#include <gz/sim/System.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/LinearVelocityCmd.hh>
#include <gz/sim/components/AngularVelocityCmd.hh>

namespace rs {
class RealSokoban final : public gz::sim::System,
    public gz::sim::ISystemConfigure, public gz::sim::ISystemPreUpdate,
    public gz::sim::ISystemPostUpdate {
  struct Command {
    long long id = 0;
    bool reset = false;
    int steps = 20;
    double vx=0, vy=0, rx=0, ry=0, bx=0, by=0, yaw=0;
  } pending, active;
  gz::transport::Node node;
  gz::transport::Node::Publisher pub;
  gz::sim::Model robot, crate;
  gz::sim::Link rl, bl;
  std::mutex mutex;
  bool queued=false, busy=false, ready=false;
  long long newest=-1;
  int remaining=0;
  std::string last;
  double rz=.09, bz=.12;
  std::chrono::steady_clock::time_point lastSend{};

  void Publish(const std::string &s) {
    gz::msgs::StringMsg m; m.set_data(s); pub.Publish(m);
  }
  void OnCommand(const gz::msgs::StringMsg &m) {
    Command c; std::string op, extra;
    std::istringstream in(m.data());
    if (!(in >> c.id >> op)) return;
    if (op == "RESET") {
      c.reset=true;
      in >> c.rx >> c.ry >> c.bx >> c.by >> c.yaw >> c.steps;
    } else if (op == "STEP") in >> c.vx >> c.vy >> c.steps;
    else {Publish("ERR " + std::to_string(c.id) + " operation"); return;}
    if (!in || (in >> extra) || c.steps < 1 || c.steps > 400 ||
        !std::isfinite(c.vx+c.vy+c.rx+c.ry+c.bx+c.by+c.yaw) ||
        std::abs(c.vx)>0.5 || std::abs(c.vy)>0.5) {
      Publish("ERR " + std::to_string(c.id) + " arguments"); return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex);
      // Retransmitted IDs must not advance physics twice.
      if (c.id <= newest) {if (!last.empty()) Publish(last); return;}
      if (busy) {Publish("ERR " + std::to_string(c.id) + " busy"); return;}
      newest=c.id; pending=c; queued=true; busy=true;
    }
    gz::msgs::WorldControl request;
    request.set_pause(true); request.set_multi_step(c.steps);
    gz::msgs::Boolean reply; bool result=false;
    const bool sent = node.Request("/world/real_sokoban/control", request,
                                  3000, reply, result);
    if (!sent || !result || !reply.data()) {
      // Client aborts the run: no unsafe retry with a new sequence ID.
      Publish("ERR " + std::to_string(c.id) + " world_control");
    }
  }
 public:
  void Configure(const gz::sim::Entity &, const std::shared_ptr<const sdf::Element> &s,
                 gz::sim::EntityComponentManager &, gz::sim::EventManager &) override {
    rz=s->Get<double>("robot_z", .09).first;
    bz=s->Get<double>("box_z", .12).first;
    pub=node.Advertise<gz::msgs::StringMsg>("/rs/state");
    node.Subscribe("/rs/command", &RealSokoban::OnCommand, this);
  }
  void PreUpdate(const gz::sim::UpdateInfo &info,
                 gz::sim::EntityComponentManager &ecm) override {
    if (!ready) {
      robot=gz::sim::Model(ecm.EntityByComponents(gz::sim::components::Model(),
                                                 gz::sim::components::Name("robot")));
      crate=gz::sim::Model(ecm.EntityByComponents(gz::sim::components::Model(),
                                                 gz::sim::components::Name("crate")));
      if (!robot.Valid(ecm) || !crate.Valid(ecm)) return;
      rl=gz::sim::Link(robot.CanonicalLink(ecm));
      bl=gz::sim::Link(crate.CanonicalLink(ecm));
      if (!rl.Valid(ecm) || !bl.Valid(ecm)) return;
      rl.EnableVelocityChecks(ecm); bl.EnableVelocityChecks(ecm); ready=true;
    }
    if (info.paused) return;
    // RS_RESET_VELOCITY_RELEASE_V1
    // Gazebo Sim 8 clears velocity commands to zero but keeps the components.
    // Remove last tick's RESET commands BEFORE accepting a new RESET. Otherwise
    // they override the robot wrench and also prevent the crate from moving.
    // A new RESET below still gets exactly one physics tick of zero velocity.
    for (const auto entity : {rl.Entity(), bl.Entity()}) {
      ecm.RemoveComponent<gz::sim::components::LinearVelocityCmd>(entity);
      ecm.RemoveComponent<gz::sim::components::AngularVelocityCmd>(entity);
    }
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (queued) {
        active=pending; remaining=active.steps; queued=false;
        if (active.reset) {
          robot.SetWorldPoseCmd(ecm, {active.rx,active.ry,rz,0,0,0});
          crate.SetWorldPoseCmd(ecm, {active.bx,active.by,bz,0,0,active.yaw});
          rl.SetLinearVelocity(ecm, {0,0,0}); rl.SetAngularVelocity(ecm, {0,0,0});
          bl.SetLinearVelocity(ecm, {0,0,0}); bl.SetAngularVelocity(ecm, {0,0,0});
        }
      }
    }
    const auto v=rl.WorldLinearVelocity(ecm).value_or(gz::math::Vector3d::Zero);
    const auto w=rl.WorldAngularVelocity(ecm).value_or(gz::math::Vector3d::Zero);
    const auto p=gz::sim::worldPose(rl.Entity(),ecm);
    gz::math::Vector3d force(120*(active.vx-v.X()),120*(active.vy-v.Y()),0);
    if (force.Length()>30) force*=30/force.Length();
    gz::math::Vector3d torque(-6*p.Rot().Roll()-1.5*w.X(),
                             -6*p.Rot().Pitch()-1.5*w.Y(),-1.5*w.Z());
    if (torque.Length()>3) torque*=3/torque.Length();
    rl.AddWorldWrench(ecm,force,torque);
  }
  void PostUpdate(const gz::sim::UpdateInfo &info,
                  const gz::sim::EntityComponentManager &ecm) override {
    if (!ready) return;
    if (!info.paused && remaining>0 && --remaining==0) {
      const auto rp=gz::sim::worldPose(rl.Entity(),ecm);
      const auto bp=gz::sim::worldPose(bl.Entity(),ecm);
      const auto rv=rl.WorldLinearVelocity(ecm).value_or(gz::math::Vector3d::Zero);
      const auto bv=bl.WorldLinearVelocity(ecm).value_or(gz::math::Vector3d::Zero);
      const auto bw=bl.WorldAngularVelocity(ecm).value_or(gz::math::Vector3d::Zero);
      std::ostringstream out; out << std::setprecision(12) << active.id << ' '
        << std::chrono::duration<double>(info.simTime).count() << ' '
        << rp.Pos().X() << ' ' << rp.Pos().Y() << ' ' << rp.Pos().Z() << ' '
        << rv.X() << ' ' << rv.Y() << ' '
        << bp.Pos().X() << ' ' << bp.Pos().Y() << ' ' << bp.Pos().Z() << ' '
        << bp.Rot().Yaw() << ' ' << bv.X() << ' ' << bv.Y() << ' ' << bw.Z() << ' '
        << rp.Rot().Roll() << ' ' << rp.Rot().Pitch() << ' '
        << bp.Rot().Roll() << ' ' << bp.Rot().Pitch();
      std::lock_guard<std::mutex> lock(mutex);
      last=out.str(); busy=false; Publish(last); lastSend=std::chrono::steady_clock::now();
    }
    // A paused-world heartbeat makes the last acknowledgement observable
    // even if DDS/Gazebo topic discovery completed just after publication.
    if (info.paused && std::chrono::steady_clock::now()-lastSend>
                       std::chrono::milliseconds(200)) {
      std::lock_guard<std::mutex> lock(mutex);
      if (!last.empty()) Publish(last);
      lastSend=std::chrono::steady_clock::now();
    }
  }
};
}
GZ_ADD_PLUGIN(rs::RealSokoban, gz::sim::System, rs::RealSokoban::ISystemConfigure,
              rs::RealSokoban::ISystemPreUpdate, rs::RealSokoban::ISystemPostUpdate)
