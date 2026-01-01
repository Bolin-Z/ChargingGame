// clang-format off
#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <deque>
#include <string>
#include <cmath>
#include <random>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <queue>
#include <execution>
#include <thread>

#include "utils.h"

using std::string, std::vector, std::deque, std::pair, std::map, std::unordered_map, std::priority_queue, std::greater, std::cout, std::endl;

// 前向声明
struct World;
struct Node;
struct Link;
struct Vehicle;

/**
 * 车辆状态枚举
 */
enum VehicleState : int {
    vsHOME = 0,  // 未出发
    vsWAIT = 1,  // 等待进入路网
    vsRUN  = 2,  // 行驶中
    vsEND  = 3   // 已到达
};

/**
 * 路径选择原则枚举
 */
enum RouteChoicePrinciple : int {
    rcpDUO = 0,    // 动态用户最优
    rcpFIXED = 1   // 固定路径
};

// -----------------------------------------------------------------------
// Node 类：交通网络节点
// -----------------------------------------------------------------------

struct Node {
    World *w;
    int id;
    string name;

    vector<Link *> in_links;   // 进入该节点的链路
    vector<Link *> out_links;  // 从该节点出发的链路

    // 刚到达此节点的车辆（尚未进入任何链路）
    vector<Vehicle *> incoming_vehicles;
    // 每个到达车辆请求的下一条链路
    vector<Link *> incoming_vehicles_requests;

    // 等待生成到出口链路的车辆队列
    deque<Vehicle *> generation_queue;

    double x;  // x坐标
    double y;  // y坐标

    // 信号灯相关
    vector<double> signal_intervals;  // 各相位时长
    double signal_offset;             // 信号偏移
    double signal_t;                  // 当前相位已持续时间
    int signal_phase;                 // 当前相位索引

    Node(
        World *w,
        const string &node_name,
        double x,
        double y,
        vector<double> signal_intervals = {0},
        double signal_offset = 0);

    void generate();       // 尝试生成一辆车进入路网
    void transfer();       // 将到达车辆转移到下一条链路
    void signal_update();  // 更新信号灯状态
};

// -----------------------------------------------------------------------
// Link 类：交通网络链路
// -----------------------------------------------------------------------

struct Link {
    World *w;

    int id;
    string name;
    double length;      // 链路长度
    Node *start_node;   // 起始节点
    Node *end_node;     // 终止节点

    double vmax;        // 自由流速度
    double delta;       // 车辆间距（1/kappa）
    double tau;         // 反应时间
    double kappa;       // 拥挤密度
    double capacity;    // 通行能力
    double backward_wave_speed;  // 后向波速
    deque<Vehicle *> vehicles;   // 链路上的车辆队列

    vector<double> traveltime_tt;  // 旅行时间记录（增量）
    vector<double> traveltime_t;   // 旅行时间记录（时间戳）

    vector<double> arrival_curve;      // 到达曲线
    vector<double> departure_curve;    // 离开曲线
    vector<double> traveltime_real;    // 实际旅行时间
    vector<double> traveltime_instant; // 瞬时旅行时间

    double merge_priority;       // 合流优先级
    double capacity_out;         // 出口通行能力
    double capacity_out_remain;  // 剩余出口通行能力

    // 信号灯相关
    vector<int> signal_group;    // 所属信号组

    Link(
        World *w,
        const string &link_name,
        const string &start_node_name,
        const string &end_node_name,
        double vmax,
        double kappa,
        double length,
        double merge_priority,
        double capacity_out=-1.0,
        vector<int> signal_group={0});

    void update();          // 更新链路状态
    void set_travel_time(); // 计算旅行时间
};

// -----------------------------------------------------------------------
// Vehicle 类：车辆
// -----------------------------------------------------------------------

struct Vehicle {
    World *w;
    int id;
    string name;

    double departure_time;  // 出发时间
    Node *orig;             // 起点
    Node *dest;             // 终点
    Link *link;             // 当前所在链路

    double arrival_time;    // 到达时间
    double travel_time;     // 旅行时间

    double x;       // 在链路上的位置
    double x_next;  // 下一时步的位置
    double v;       // 当前速度

    Vehicle *leader;    // 前车
    Vehicle *follower;  // 后车

    int state;  // 车辆状态：vsHOME=0, vsWAIT=1, vsRUN=2, vsEND=3

    double arrival_time_link;  // 到达当前链路的时间

    // 路径选择相关
    Link *route_next_link;           // 下一条链路
    int route_choice_flag_on_link;   // 是否已在当前链路上做过路径选择
    double route_adaptive;           // 自适应路径选择参数
    double route_choice_uncertainty; // 路径选择不确定性
    map<Link *, double> route_preference;  // 链路偏好（目的地->链路->偏好值）
    int route_choice_principle;      // 路径选择原则
    vector<Link *> links_preferred;  // 偏好链路列表（DUO模式用）

    // ========== 新增：预定路径支持 ==========
    vector<Link *> predefined_route;  // 有序预定路径列表
    size_t route_index;               // 当前执行到第几个链路
    bool use_predefined_route;        // 是否使用预定路径模式

    // 日志记录（每时步）
    vector<double> log_t;      // 时间日志
    vector<int> log_state;     // 状态日志
    vector<int> log_link;      // 链路ID日志
    vector<double> log_x;      // 位置日志
    vector<double> log_v;      // 速度日志

    // 路径级日志（仅链路变化时记录）
    // 对应 UXsim 1.8.2 uxsim.py:975
    // 格式：pair<时步数, 链路指针>
    // nullptr 表示 "home"（第一条）或 "end"（最后一条）
    vector<pair<int, Link*>> log_t_link;
    Link* link_old;  // 用于检测链路变化，对应 uxsim.py:976

    Vehicle(
        World *w,
        const string &vehicle_name,
        double departure_time,
        const string &orig_name,
        const string &dest_name);

    void update();              // 更新车辆状态
    void end_trip();            // 结束行程
    void car_follow_newell();   // Newell跟驰模型
    void route_next_link_choice(vector<Link*> linkset);  // 选择下一条链路
    void record_travel_time(Link *link, double t);       // 记录旅行时间
    void log_data();            // 记录日志数据

    // ========== 新增方法 ==========
    void assign_route(vector<Link*> route);              // 分配预定路径（Link指针列表）
    void assign_route_by_name(vector<string> route_names); // 分配预定路径（链路名称列表）
};

// -----------------------------------------------------------------------
// World 类：仿真世界
// -----------------------------------------------------------------------

struct World {
    // 仿真配置
    long long timestamp;
    string name;

    double t_max;              // 仿真总时长
    double delta_n;            // 车队大小（每个Vehicle对象代表的实际车辆数）
    double tau;                // 反应时间
    double duo_update_time;    // DUO更新时间间隔
    double duo_update_weight;  // DUO更新权重
    int print_mode;            // 打印模式

    double delta_t;            // 时步长度
    size_t total_timesteps;    // 总时步数
    size_t timestep_for_route_update;  // 路径更新间隔（时步数）

    int node_id;     // 节点ID计数器
    int link_id;     // 链路ID计数器
    int vehicle_id;  // 车辆ID计数器

    // 对象集合
    vector<Vehicle *> vehicles;         // 所有车辆
    vector<Link *> links;               // 所有链路
    vector<Node *> nodes;               // 所有节点
    unordered_map<int, Vehicle *> vehicles_living;   // 存活车辆（HOME/WAIT/RUN状态）
    unordered_map<int, Vehicle *> vehicles_running;  // 行驶中车辆（RUN状态）
    unordered_map<string, Node *> nodes_map;         // 节点名->节点指针
    unordered_map<string, Link *> links_map;         // 链路名->链路指针
    unordered_map<string, Vehicle *> vehicles_map;   // 车辆名->车辆指针

    size_t timestep;  // 当前时步
    double time;      // 当前仿真时间（秒）

    double route_adaptive;
    double route_choice_uncertainty;
    vector<map<Link *, double>> route_preference;  // route_preference[dest_id][link]: 到目的地dest的链路link偏好值

    // 邻接矩阵
    vector<vector<int>> adj_mat;        // 邻接矩阵
    vector<vector<double>> adj_mat_time; // 时间邻接矩阵
    vector<vector<int>> route_next;     // 最短路径下一跳
    vector<vector<double>> route_dist;  // 最短路径距离

    bool flag_initialized;  // 是否已初始化

    // 统计数据
    double ave_v;           // 平均速度
    double ave_vratio;      // 平均速度比
    double trips_total;     // 总行程数
    double trips_completed; // 完成行程数

    // 随机数
    long long random_seed;
    std::mt19937 rng;

    std::ostream *writer;  // 输出流

    World(
        const string &world_name,
        double t_max,
        double delta_n,
        double tau,
        double duo_update_time,
        double duo_update_weight,
        double route_choice_uncertainty,
        int print_mode,
        long long random_seed,
        bool vehicle_log_mode);

    // 析构函数：释放所有动态分配的对象
    ~World();

    void initialize_adj_matrix();    // 初始化邻接矩阵
    void update_adj_time_matrix();   // 更新时间邻接矩阵

    void route_choice_duo();         // DUO路径选择更新

    // 全源最短路径搜索（Dijkstra）
    pair<vector<vector<double>>, vector<vector<int>>>
        route_search_all(const vector<vector<double>> &adj, double infty);

    void print_scenario_stats();     // 打印场景统计信息
    void print_simple_results();     // 打印简单结果
    void main_loop(double duration_t, double end_t);  // 主仿真循环

    bool check_simulation_ongoing(); // 检查仿真是否继续

    Node *get_node(const string &node_name);        // 根据名称获取节点
    Link *get_link(const string &link_name);        // 根据名称获取链路
    Link *get_link_by_id(const int link_id);        // 根据ID获取链路
    Vehicle *get_vehicle(const string &vehicle_name); // 根据名称获取车辆

    // 批量分配预定路径（性能优化：减少 Python-C++ 边界跨越）
    void batch_assign_routes(const vector<pair<int, vector<string>>> &assignments);

    size_t vehicle_log_reserve_size;  // 车辆日志预留大小
    bool vehicle_log_mode;            // 是否记录车辆日志
};
