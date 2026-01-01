// clang-format off

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <memory>
#include <sstream>
#include <string>
#include <iostream>
#include <streambuf>

#include "traffic.cpp"

namespace py = pybind11;


// ----------------------------------------------------------------------
// 自定义 streambuf: 将输出重定向到 Python 的 sys.stdout
// ----------------------------------------------------------------------
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif
class py_stdout_redirect_buf : public std::streambuf {
public:
    py_stdout_redirect_buf() {
        py::object sys = py::module_::import("sys");
        py_stdout = sys.attr("stdout");
    }
protected:
    // 单字符输出
    virtual int overflow(int c) override {
        if (c != EOF) {
            std::string s(1, static_cast<char>(c));
            py_stdout.attr("write")(s);
        }
        return c;
    }
    // 多字符输出
    virtual std::streamsize xsputn(const char* s, std::streamsize n) override {
        std::string str(s, n);
        py_stdout.attr("write")(str);
        return n;
    }
private:
    py::object py_stdout;
};
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

// ----------------------------------------------------------------------
// 获取 Python stdout 输出流
// ----------------------------------------------------------------------
std::ostream* get_pyout() {
    static py_stdout_redirect_buf custom_buf;
    static std::ostream pyout(&custom_buf);
    return &pyout;
}

// ----------------------------------------------------------------------
// create_world() 实现
// ----------------------------------------------------------------------
std::unique_ptr<World> create_world(
        const std::string &world_name,
        double t_max,
        double delta_n,
        double tau,
        double duo_update_time,
        double duo_update_weight,
        double route_choice_uncertainty,
        int print_mode,
        long long random_seed,
        bool vehicle_log_mode){
    auto world = std::make_unique<World>(
        world_name,
        t_max,
        delta_n,
        tau,
        duo_update_time,
        duo_update_weight,
        route_choice_uncertainty,
        print_mode,
        random_seed,
        vehicle_log_mode);
    // 将输出重定向到 Python sys.stdout
    world->writer = get_pyout();
    return world;
}

// ----------------------------------------------------------------------
// 场景定义函数
// ----------------------------------------------------------------------
void add_node(World &world, const std::string &node_name, double x, double y,
        vector<double> signal_intervals = {0}, double signal_offset = 0) {
    new Node(&world, node_name, x, y, signal_intervals, signal_offset);
}

void add_link(
        World &world,
        const std::string &link_name,
        const std::string &start_node_name,
        const std::string &end_node_name,
        double vmax,
        double kappa,
        double length,
        double merge_priority,
        double capacity_out,
        vector<int> signal_group={0}){
    new Link(&world, link_name, start_node_name, end_node_name,
                        vmax, kappa, length, merge_priority, capacity_out, signal_group);
}

// add_demand 声明（实现在 traffic.cpp 中）
void add_demand(
    World *w,
    const std::string &orig_name,
    const std::string &dest_name,
    double start_t,
    double end_t,
    double flow,
    vector<string> links_preferred_str,
    bool use_predefined_route);

// ----------------------------------------------------------------------
// 返回编译日期时间
// ----------------------------------------------------------------------
std::string get_compile_datetime() {
    return std::string("Compiled on ") + __DATE__ + " at " + __TIME__;
}

// ----------------------------------------------------------------------
// Pybind11 模块定义
// ----------------------------------------------------------------------
PYBIND11_MODULE(trafficppy, m) {
    m.doc() = "trafficppy: pybind11 bindings for C++ mesoscopic traffic simulation with predefined route support";

    //
    // create_world
    //
    m.def("create_world", &create_world,
          py::arg("world_name"),
          py::arg("t_max"),
          py::arg("delta_n"),
          py::arg("tau"),
          py::arg("duo_update_time"),
          py::arg("duo_update_weight"),
          py::arg("route_choice_uncertainty"),
          py::arg("print_mode"),
          py::arg("random_seed"),
          py::arg("vehicle_log_mode"),
          R"docstring(
          Create a World (simulation environment).

          Parameters
          ----------
          world_name : str
              The name of the world.
          t_max : float
              The simulation duration.
          delta_n : float
              The platoon size.
          tau : float
              The reaction time.
          duo_update_time : float
              The time interval for route choice update.
          duo_update_weight : float
              The update weight for route choice.
          route_choice_uncertainty : float
              The noise in route choice.
          print_mode : int
              Whether print the simulation progress or not.
          random_seed : int
              The random seed.
          vehicle_log_mode : bool
              Whether save vehicle data or not.

          Returns
          -------
          World
              World simulation object.
          )docstring");

    //
    // add_node
    //
    m.def("add_node", &add_node,
          py::arg("world"),
          py::arg("node_name"),
          py::arg("x"),
          py::arg("y"),
          py::arg("signal_intervals"),
          py::arg("signal_offset"),
          R"docstring(
          Add a node to the world.

          Parameters
          ----------
          world : World
              The world to which the node belongs.
          node_name : str
              The name of the node.
          x : float
              The x-coordinate of the node.
          y : float
              The y-coordinate of the node.
          signal_intervals : list of float
              A list representing the signal at the node.
          signal_offset : float
              The offset of the signal.
          )docstring");

    //
    // add_link
    //
    m.def("add_link", &add_link,
          py::arg("world"),
          py::arg("link_name"),
          py::arg("start_node_name"),
          py::arg("end_node_name"),
          py::arg("vmax"),
          py::arg("kappa"),
          py::arg("length"),
          py::arg("merge_priority"),
          py::arg("capacity_out"),
          py::arg("signal_group"),
          R"docstring(
          Add a link to the world.

          Parameters
          ----------
          world : World
              The world to which the link belongs.
          link_name : str
              The name of the link.
          start_node_name : str
              The name of the start node.
          end_node_name : str
              The name of the end node.
          vmax : float
              The free flow speed on the link.
          kappa : float
              The jam density on the link.
          length : float
              The length of the link.
          merge_priority : float
              The priority of the link when merging.
          capacity_out : float
              The capacity out of the link.
          signal_group : list of int
              The signal group(s) to which the link belongs.
          )docstring");

    //
    // add_demand (新增 use_predefined_route 参数)
    //
    m.def("add_demand", &add_demand,
          py::arg("world"),
          py::arg("orig_name"),
          py::arg("dest_name"),
          py::arg("start_t"),
          py::arg("end_t"),
          py::arg("flow"),
          py::arg("links_preferred_str"),
          py::arg("use_predefined_route") = false,
          R"docstring(
          Add demand (vehicle generation) to the world.

          Parameters
          ----------
          world : World
              The world to which the demand belongs.
          orig_name : str
              The origin node.
          dest_name : str
              The destination node.
          start_t : float
              The start time of demand.
          end_t : float
              The end time of demand.
          flow : float
              The flow rate of vehicles.
          links_preferred_str : list of str
              The names of the links (preferred links for DUO mode, or predefined route).
          use_predefined_route : bool, optional
              If True, use predefined route mode. Default is False.
          )docstring");

    //
    // World 类 (添加 py::dynamic_attr() 支持 Python 层动态属性如 _pending_route_assignments)
    //
    py::class_<World>(m, "World", py::dynamic_attr())
        .def("initialize_adj_matrix", &World::initialize_adj_matrix)
        .def("print_scenario_stats", &World::print_scenario_stats)
        .def("main_loop", &World::main_loop)
        .def("check_simulation_ongoing", &World::check_simulation_ongoing)
        .def("print_simple_results", &World::print_simple_results)
        .def("release", &World::release, "显式释放资源，在创建新World前调用以避免GC延迟导致的堆冲突")
        .def("update_adj_time_matrix", &World::update_adj_time_matrix)
        .def("get_node", &World::get_node,
             py::return_value_policy::reference,
             "Get a Node by name (reference)")
        .def("get_link", &World::get_link,
             py::return_value_policy::reference,
             "Get a Link by name (reference)")
        .def("get_vehicle", &World::get_vehicle,
             py::return_value_policy::reference,
             "Get a Vehicle by name (reference)")
        .def("batch_assign_routes", &World::batch_assign_routes,
             py::arg("assignments"),
             "Batch assign predefined routes to vehicles (performance optimization)")
        // 使用 def_property_readonly 并指定 reference_internal 策略
        // 确保返回的 vector 中的指针正确引用 C++ 对象
        .def_property_readonly("VEHICLES", [](World &w) -> std::vector<Vehicle*>& {
            return w.vehicles;
        }, py::return_value_policy::reference_internal,
           "Vector of pointers to all Vehicles in the world.")
        .def_property_readonly("LINKS", [](World &w) -> std::vector<Link*>& {
            return w.links;
        }, py::return_value_policy::reference_internal,
           "Vector of pointers to all Links in the world.")
        .def_property_readonly("NODES", [](World &w) -> std::vector<Node*>& {
            return w.nodes;
        }, py::return_value_policy::reference_internal,
           "Vector of pointers to all Nodes in the world.")
        .def_readonly("timestep", &World::timestep)
        .def_readonly("time", &World::time)
        .def_readonly("delta_t", &World::delta_t)
        .def_readonly("DELTAT", &World::delta_t)
        .def_readonly("t_max", &World::t_max)
        .def_readonly("TMAX", &World::t_max)
        .def_readonly("name", &World::name)
        .def_readonly("deltan", &World::delta_n)
        ;

    //
    // Node 类
    // 使用 py::nodelete 防止 pybind11 在 GC 时 delete Node
    // Node 的生命周期完全由 World 管理
    //
    py::class_<Node, std::unique_ptr<Node, py::nodelete>>(m, "Node")
        .def(py::init<World *, const std::string &, double, double>(),
             py::arg("world"),
             py::arg("node_name"),
             py::arg("x"),
             py::arg("y"))
        .def_readonly("W", &Node::w)
        .def_readonly("id", &Node::id)
        .def_readonly("name", &Node::name)
        .def_readwrite("x", &Node::x)
        .def_readwrite("y", &Node::y)
        .def_readwrite("signal_intervals", &Node::signal_intervals)
        .def_readwrite("signal_offset", &Node::signal_offset)
        .def_readwrite("signal_t", &Node::signal_t)
        .def_readwrite("signal_phase", &Node::signal_phase)
        .def_readonly("in_links", &Node::in_links)
        .def_readonly("out_links", &Node::out_links)
        .def_readonly("incoming_vehicles", &Node::incoming_vehicles)
        .def_readonly("generation_queue", &Node::generation_queue)
        .def("generate", &Node::generate)
        .def("transfer", &Node::transfer)
        ;

    //
    // Link 类
    // 使用 py::nodelete 防止 pybind11 在 GC 时 delete Link
    // Link 的生命周期完全由 World 管理
    //
    py::class_<Link, std::unique_ptr<Link, py::nodelete>>(m, "Link", py::dynamic_attr())
        .def(py::init<World *, const std::string &, const std::string &, const std::string &,
                      double, double, double, double, double>(),
             py::arg("world"),
             py::arg("link_name"),
             py::arg("start_node_name"),
             py::arg("end_node_name"),
             py::arg("vmax"),
             py::arg("kappa"),
             py::arg("length"),
             py::arg("merge_priority"),
             py::arg("capacity_out"))
        .def_readonly("W", &Link::w)
        .def_readonly("id", &Link::id)
        .def_readonly("name", &Link::name)
        .def_readwrite("length", &Link::length)
        .def_readwrite("u", &Link::vmax)
        .def_readwrite("vmax", &Link::vmax)
        .def_readwrite("kappa", &Link::kappa)
        .def_readwrite("delta", &Link::delta)
        .def_readwrite("tau", &Link::tau)
        .def_readwrite("capacity", &Link::capacity)
        .def_readwrite("w", &Link::backward_wave_speed)
        .def_readwrite("merge_priority", &Link::merge_priority)
        .def_readwrite("capacity_out", &Link::capacity_out)
        .def_readwrite("signal_group", &Link::signal_group)
        .def_readonly("start_node", &Link::start_node)
        .def_readonly("end_node", &Link::end_node)
        .def_readonly("vehicles", &Link::vehicles)
        .def_readonly("arrival_curve", &Link::arrival_curve)
        .def_readonly("cum_arrival", &Link::arrival_curve)
        .def_readonly("departure_curve", &Link::departure_curve)
        .def_readonly("cum_departure", &Link::departure_curve)
        .def_readonly("traveltime_real", &Link::traveltime_real)
        .def_readonly("traveltime_instant", &Link::traveltime_instant)
        .def("update", &Link::update)
        .def("set_travel_time", &Link::set_travel_time)
        ;

    //
    // Vehicle 类
    // 使用 py::nodelete 防止 pybind11 在 GC 时 delete Vehicle
    // Vehicle 的生命周期完全由 World 管理（通过 World::release() 或 ~World() 释放）
    //
    py::class_<Vehicle, std::unique_ptr<Vehicle, py::nodelete>>(m, "Vehicle", py::dynamic_attr())
        .def(py::init<World *, const std::string &, double, const std::string &, const std::string &>(),
             py::arg("world"),
             py::arg("name"),
             py::arg("departure_time"),
             py::arg("orig_name"),
             py::arg("dest_name"))
        .def_readonly("W", &Vehicle::w)
        .def_readonly("id", &Vehicle::id)
        .def_readwrite("name", &Vehicle::name)
        .def_readonly("departure_time", &Vehicle::departure_time)
        .def_readwrite("orig", &Vehicle::orig)
        .def_readwrite("dest", &Vehicle::dest)
        .def_readonly("link", &Vehicle::link)
        .def_readonly("x", &Vehicle::x)
        .def_readonly("x_next", &Vehicle::x_next)
        .def_readonly("v", &Vehicle::v)
        .def_readonly("leader", &Vehicle::leader)
        .def_readonly("follower", &Vehicle::follower)
        .def_readonly("state", &Vehicle::state)
        .def_readonly("arrival_time_link", &Vehicle::arrival_time_link)
        .def_readwrite("route_next_link", &Vehicle::route_next_link)
        .def_readwrite("route_choice_flag_on_link", &Vehicle::route_choice_flag_on_link)
        .def_readwrite("route_adaptive", &Vehicle::route_adaptive)
        .def_readwrite("route_preference", &Vehicle::route_preference)
        .def_readwrite("links_preferred", &Vehicle::links_preferred)
        // ========== 新增：预定路径属性 ==========
        .def_readwrite("predefined_route", &Vehicle::predefined_route)
        .def_readwrite("route_index", &Vehicle::route_index)
        .def_readwrite("use_predefined_route", &Vehicle::use_predefined_route)
        .def("assign_route", &Vehicle::assign_route,
             py::arg("route"),
             "Assign a predefined route (list of Link pointers)")
        .def("assign_route_by_name", &Vehicle::assign_route_by_name,
             py::arg("route_names"),
             "Assign a predefined route by link names")
        // ========== 日志属性 ==========
        .def_readonly("log_t", &Vehicle::log_t)
        .def_readonly("log_state", &Vehicle::log_state)
        .def_readonly("log_link", &Vehicle::log_link)
        .def_readonly("log_x", &Vehicle::log_x)
        .def_readonly("log_v", &Vehicle::log_v)
        // 路径级日志，对应 UXsim 1.8.2 的 log_t_link
        // 返回 list[tuple[int, Link | None]]
        // None 表示 "home"（第一条）或 "end"（最后一条）
        .def_readonly("log_t_link", &Vehicle::log_t_link)
        .def_readonly("arrival_time", &Vehicle::arrival_time)
        .def_readonly("travel_time", &Vehicle::travel_time)
        ;

    m.def("get_compile_datetime", &get_compile_datetime, "Return the compile date and time");
}
