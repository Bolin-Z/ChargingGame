// clang-format off

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <deque>
#include <string>
#include <cmath>
#include <random>
#include <map>
#include <algorithm>
#include <chrono>
#include <queue>

#include "traffic.h"

using std::string, std::vector, std::deque, std::pair, std::map, std::unordered_map;
using std::round, std::floor, std::ceil;
using std::cout, std::endl;

// -----------------------------------------------------------------------
// Node 实现
// -----------------------------------------------------------------------

/**
 * 创建节点
 */
Node::Node(World *w, const string &node_name, double x, double y, vector<double> signal_intervals, double signal_offset)
    : w(w),
      id(w->node_id),
      name(node_name),
      x(x),
      y(y),
      signal_intervals(signal_intervals),
      signal_offset(signal_offset){
    w->nodes.push_back(this);
    w->node_id++;
    w->nodes_map[node_name] = this;

    signal_t = signal_offset;
    signal_phase = 0;
}

/**
 * 尝试生成一辆车进入路网
 */
void Node::generate(){
    if (!generation_queue.empty()){
        Vehicle *veh = generation_queue.front();

        // 选择出口链路
        veh->route_next_link_choice(out_links);

        if (!out_links.empty() && veh->route_next_link != nullptr){
            Link *outlink = veh->route_next_link;

            // 检查出口链路是否能接收新车辆
            if (outlink->vehicles.empty() || outlink->vehicles.back()->x > outlink->delta * w->delta_n){
                generation_queue.pop_front();

                veh->state = vsRUN;
                veh->link = outlink;
                veh->x = 0.0;
                veh->record_travel_time(nullptr, (double)w->timestep * w->delta_t);

                w->vehicles_running[veh->id] = veh;

                // 设置前后车关系
                if (!outlink->vehicles.empty()){
                    veh->leader = outlink->vehicles.back();
                    outlink->vehicles.back()->follower = veh;
                }
                outlink->vehicles.push_back(veh);

                // 更新到达曲线
                outlink->arrival_curve[w->timestep] += w->delta_n;
            }
        }
    }
}

/**
 * 更新信号灯状态
 */
void Node::signal_update(){
    if (signal_intervals.size() > 1){
        while (signal_t > signal_intervals[signal_phase]){
            signal_t -= signal_intervals[signal_phase];
            signal_phase++;
            if (signal_phase >= (int)signal_intervals.size()){
                signal_phase = 0;
            }
        }
        signal_t += w->delta_t;
    }
}

/**
 * 将到达车辆转移到下一条链路
 */
void Node::transfer(){
    for (auto outlink : out_links){
        if (outlink->vehicles.empty() ||
            outlink->vehicles.back()->x > outlink->delta * w->delta_n){

            // 收集想要进入该出口链路的合流车辆
            vector<Vehicle *> merging_vehs;
            vector<double> merge_priorities;
            for (auto veh : incoming_vehicles){
                if (veh->route_next_link == outlink &&
                        veh->link->capacity_out_remain >= w->delta_n &&
                        contains(veh->link->signal_group, signal_phase)){
                    merging_vehs.push_back(veh);
                    if (veh->link){
                        merge_priorities.push_back(veh->link->merge_priority);
                    }else{
                        merge_priorities.push_back(1.0);
                    }
                }
            }
            if (merging_vehs.empty()){
                continue;
            }

            // 根据优先级随机选择一辆车进行合流
            Vehicle *chosen_veh = random_choice<Vehicle>(
                merging_vehs,
                merge_priorities,
                w->rng);
            if (!chosen_veh){
                continue;
            }

            chosen_veh->link->capacity_out_remain -= w->delta_n;

            // 更新离开曲线和到达曲线
            chosen_veh->link->departure_curve[w->timestep] += w->delta_n;
            outlink->arrival_curve[w->timestep] += w->delta_n;

            // 记录旅行时间
            chosen_veh->record_travel_time(chosen_veh->link, (double)w->timestep * w->delta_t);

            // 从原链路移除
            chosen_veh->link->vehicles.pop_front();

            chosen_veh->link = outlink;
            chosen_veh->x = 0.0;
            chosen_veh->x_next = 0.0;

            // 更新前后车关系
            if (chosen_veh->follower){
                chosen_veh->follower->leader = nullptr;
            }
            chosen_veh->leader = nullptr;
            chosen_veh->follower = nullptr;

            if (!outlink->vehicles.empty()){
                Vehicle *leader_veh = outlink->vehicles.back();
                chosen_veh->leader = leader_veh;
                leader_veh->follower = chosen_veh;
            }
            outlink->vehicles.push_back(chosen_veh);

            // 从到达车辆列表中移除
            remove_from_vector(incoming_vehicles, chosen_veh);
        }
    }

    incoming_vehicles.clear();
    incoming_vehicles_requests.clear();
}

// -----------------------------------------------------------------------
// Link 实现
// -----------------------------------------------------------------------

/**
 * 创建链路
 */
Link::Link(
    World *w,
    const string &link_name,
    const string &start_node_name,
    const string &end_node_name,
    double vmax,
    double kappa,
    double length,
    double merge_priority,
    double capacity_out,
    vector<int> signal_group)
    : w(w),
      id(w->link_id),
      name(link_name),
      length(length),
      vmax(vmax),
      kappa(kappa),
      merge_priority(merge_priority),
      capacity_out(capacity_out),
      signal_group(signal_group){

    if (kappa <= 0.0){
        kappa = 0.2;
    }
    delta = 1.0/kappa;
    tau = w->tau;

    backward_wave_speed = 1/tau/kappa;
    capacity = vmax*backward_wave_speed*kappa/(vmax+backward_wave_speed);

    if (capacity_out < 0.0){
        capacity_out = 10e10;
    }
    capacity_out_remain = capacity_out*w->delta_t;

    start_node = w->nodes_map[start_node_name];
    end_node = w->nodes_map[end_node_name];

    arrival_curve.resize(w->total_timesteps, 0.0);
    departure_curve.resize(w->total_timesteps, 0.0);

    traveltime_real.resize(w->total_timesteps, 0.0);
    traveltime_instant.resize(w->total_timesteps, 0.0);

    // 将自身加入节点的链路列表
    start_node->out_links.push_back(this);
    end_node->in_links.push_back(this);

    w->links.push_back(this);
    w->link_id++;
    w->links_map[link_name] = this;
}

/**
 * 更新链路状态
 */
void Link::update(){
    set_travel_time();

    if (w->timestep != 0){
        arrival_curve[w->timestep] = arrival_curve[w->timestep-1];
        departure_curve[w->timestep] = departure_curve[w->timestep-1];
    }

    if (capacity_out < 10e9){
        if (capacity_out_remain < w->delta_n){
            capacity_out_remain += capacity_out*w->delta_t;
        }
    } else {
        capacity_out_remain = 10e9;
    }
}

/**
 * 计算旅行时间
 */
void Link::set_travel_time(){
    // 最后一辆车的实际旅行时间
    if (!traveltime_tt.empty() && !vehicles.empty()){
        traveltime_real[w->timestep] = (double)traveltime_tt.back();
    }else{
        traveltime_real[w->timestep] = (double)length / (double)vmax;
    }

    // 瞬时旅行时间 = 长度 / 平均速度
    if (!vehicles.empty()){
        double vsum = 0.0;
        for (auto veh : vehicles){
            vsum += veh->v;
        }
        double avg_v = vsum / (double)vehicles.size();
        if (avg_v > vmax / 10.0){
            traveltime_instant[w->timestep] = (double)length / avg_v;
        }else{
            traveltime_instant[w->timestep] = (double)length / (vmax / 10.0);
        }
    }else{
        traveltime_instant[w->timestep] = (double)length / (double)vmax;
    }
}

// -----------------------------------------------------------------------
// Vehicle 实现
// -----------------------------------------------------------------------

/**
 * 创建车辆
 */
Vehicle::Vehicle(
    World *w,
    const string &vehicle_name,
    double departure_time,
    const string &orig_name,
    const string &dest_name)
    : w(w),
      id(w->vehicle_id),
      name(vehicle_name),
      departure_time(departure_time),
      orig(nullptr),
      dest(nullptr),
      link(nullptr),
      x(0.0),
      x_next(0.0),
      v(0.0),
      leader(nullptr),
      follower(nullptr),
      state(vsHOME),
      arrival_time(0.0),
      travel_time(0.0),
      arrival_time_link(0.0),
      route_next_link(nullptr),
      route_choice_flag_on_link(0),
      route_choice_principle(rcpDUO),
      route_adaptive(0.0),
      route_choice_uncertainty(0.0),
      // ========== 新增：预定路径属性初始化 ==========
      route_index(0),
      use_predefined_route(false),
      // ========== 新增：路径级日志初始化 ==========
      link_old(nullptr){
    orig = w->nodes_map[orig_name];
    dest = w->nodes_map[dest_name];

    // 初始化链路偏好
    for (auto ln : w->links){
        route_preference[ln] = 0.0;
    }
    route_choice_uncertainty = w->route_choice_uncertainty;

    // 初始化路径级日志，对应 UXsim 1.8.2 uxsim.py:975
    // [[int(s.departure_time*s.W.DELTAT), "home"]]
    int departure_timestep = static_cast<int>(departure_time / w->delta_t);
    log_t_link.push_back({departure_timestep, nullptr});  // nullptr 表示 "home"

    log_t.reserve(w->vehicle_log_reserve_size);
    log_state.reserve(w->vehicle_log_reserve_size);
    log_link.reserve(w->vehicle_log_reserve_size);
    log_x.reserve(w->vehicle_log_reserve_size);
    log_v.reserve(w->vehicle_log_reserve_size);

    w->vehicles.push_back(this);
    w->vehicles_living[id] = this;
    w->vehicle_id++;
    w->vehicles_map[vehicle_name] = this;
}

/**
 * 更新车辆状态
 */
void Vehicle::update(){

    if (state == vsHOME){
        if ((double)w->timestep * w->delta_t >= departure_time){
            log_data();
            state = vsWAIT;
            orig->generation_queue.push_back(this);
        }
    }else if (state == vsWAIT){
        log_data();
    }else if (state == vsRUN){
        log_data();

        if (x == 0.0){
            route_choice_flag_on_link = 0;
        }

        // 更新速度和位置
        v = (x_next - x) / (w->delta_t);
        x = x_next;

        // 检查是否到达链路末端
        if (std::fabs(x - link->length) < 1e-9){
            // ========== 修改：预定路径完成判断 ==========
            bool trip_complete = false;

            if (use_predefined_route) {
                // 预定路径模式：先调用 route_next_link_choice 更新 route_index
                route_next_link_choice(link->end_node->out_links);

                // 检查是否已完成所有预定路径
                if (route_index >= predefined_route.size()) {
                    trip_complete = true;
                }
            } else {
                // DUO模式：用终点节点判断
                if (link->end_node == dest) {
                    trip_complete = true;
                }
            }

            if (trip_complete) {
                end_trip();
                log_data();
            } else {
                // 预定路径模式已在上面调用过 route_next_link_choice
                // DUO模式需要在这里调用
                if (!use_predefined_route) {
                    route_next_link_choice(link->end_node->out_links);
                }
                link->end_node->incoming_vehicles.push_back(this);
                link->end_node->incoming_vehicles_requests.push_back(route_next_link);
            }
        }
    }else if (state == vsEND){
        // 不做任何事
    }
}

/**
 * 结束行程
 */
void Vehicle::end_trip(){
    state = vsEND;
    link->departure_curve[w->timestep] += w->delta_n;
    record_travel_time(link, (double)w->timestep * w->delta_t);

    arrival_time = (double)w->timestep * w->delta_t;
    travel_time = arrival_time - departure_time;

    w->vehicles_living.erase(id);
    w->vehicles_running.erase(id);

    link->vehicles.pop_front();

    if (follower){
        follower->leader = nullptr;
    }
    link = nullptr;
    x = 0.0;
}

/**
 * Newell跟驰模型
 */
void Vehicle::car_follow_newell(){
    // 自由流
    x_next = x + link->vmax * w->delta_t;

    // 拥挤流
    if (leader != nullptr){
        double gap = leader->x - link->delta * w->delta_n;
        if (x_next >= gap){
            x_next = gap;
        }
    }

    // 位置不能后退
    if (x_next < x){
        x_next = x;
    }

    // 不能超过链路长度
    if (x_next >= link->length){
        x_next = link->length;
    }
}

/**
 * 选择下一条链路
 */
void Vehicle::route_next_link_choice(vector<Link*> linkset){
    // ========== 新增：预定路径模式 ==========
    if (use_predefined_route && !predefined_route.empty()) {
        // 转移确认机制：确认已成功进入预期链路后才递增索引
        // 核心逻辑：只有当车辆实际进入了 route_next_link 指向的链路时，才认为这一步路径执行完成
        if (route_next_link != nullptr &&
            link != nullptr &&
            route_next_link == link) {
            route_index++;
        }

        // 检查路径是否已完成
        if (route_index >= predefined_route.size()) {
            route_next_link = nullptr;
            route_choice_flag_on_link = 1;
            return;
        }

        // 设置下一个预定链路作为目标
        route_next_link = predefined_route[route_index];
        route_choice_flag_on_link = 1;
        return;
    }

    // ========== 原有逻辑（DUO模式） ==========
    if (linkset.empty()){
        route_next_link = nullptr;
        route_choice_flag_on_link = 1;
        return;
    }

    vector<double> outlink_pref;
    bool prefer_flag = 0;

    if (!links_preferred.empty()) {
        for (auto ln_out : linkset){
            outlink_pref.push_back(0);
            for (auto ln_prefer : links_preferred){
                if (ln_out == ln_prefer){
                    outlink_pref.back() = 1;
                    prefer_flag = 1;
                }
            }
        }
    }
    if (prefer_flag == 0){
        // 没有指定偏好链路，使用正常的路径选择
        outlink_pref = {};
        for (auto ln : linkset){
            outlink_pref.push_back(w->route_preference[dest->id][ln]);
        }
    }

    route_next_link = random_choice<Link>(
        linkset,
        outlink_pref,
        w->rng);
    route_choice_flag_on_link = 1;
}

/**
 * 记录旅行时间
 * 同时更新 traveltime_real，覆盖从进入时刻到末尾的所有时步
 * 对应 UXsim 1.8.2 uxsim.py:287 和 uxsim.py:1083 的逻辑
 */
void Vehicle::record_travel_time(Link *link, double t){
    if (link != nullptr){
        link->traveltime_t.push_back(t);
        double actual_tt = t - arrival_time_link;
        link->traveltime_tt.push_back(actual_tt);

        // 更新 traveltime_real：从进入时刻到末尾的所有时步
        // 对应 UXsim: traveltime_actual[int(link_arrival_time/DELTAT):] = actual_tt
        int arrival_timestep = (int)(arrival_time_link / w->delta_t);
        if (arrival_timestep < 0) arrival_timestep = 0;
        for (size_t i = (size_t)arrival_timestep; i < link->traveltime_real.size(); i++) {
            link->traveltime_real[i] = actual_tt;
        }
    }
    arrival_time_link = t + 1.0;
}

/**
 * 记录日志数据
 */
void Vehicle::log_data(){
    if (w->vehicle_log_mode == 1){
        log_t.push_back((double)w->timestep * w->delta_t);
        log_state.push_back(state);
        if (link) {
            log_link.push_back(link->id);
        } else {
            log_link.push_back(-1);
        }
        log_x.push_back(x);
        if (link != nullptr && std::fabs(x - (link->length - 1.0)) > 1e-9){
            log_v.push_back(v);
        } else {
            log_v.push_back(0.0);
        }
    }

    // ========== 新增：路径级日志记录 ==========
    // 对应 UXsim 1.8.2 uxsim.py:1383-1388
    // 仅在链路变化时记录
    if (link != link_old) {
        if (state == vsRUN) {
            log_t_link.push_back({static_cast<int>(w->timestep), link});
        } else if (state == vsEND) {
            log_t_link.push_back({static_cast<int>(w->timestep), nullptr});  // nullptr 表示 "end"
        }
        link_old = link;
    }
}

// ========== 新增方法：分配预定路径 ==========

/**
 * 分配预定路径（Link指针列表）
 */
void Vehicle::assign_route(vector<Link*> route) {
    predefined_route = route;
    route_index = 0;
    use_predefined_route = true;

    // 初始化第一个链路作为目标
    if (!predefined_route.empty()) {
        route_next_link = predefined_route[0];
    }
}

/**
 * 分配预定路径（链路名称列表）
 */
void Vehicle::assign_route_by_name(vector<string> route_names) {
    predefined_route.clear();
    for (const auto& name : route_names) {
        Link* ln = w->links_map[name];
        if (ln != nullptr) {
            predefined_route.push_back(ln);
        }
    }
    route_index = 0;
    use_predefined_route = true;

    if (!predefined_route.empty()) {
        route_next_link = predefined_route[0];
    }
}

// -----------------------------------------------------------------------
// World 实现
// -----------------------------------------------------------------------

/**
 * 创建仿真世界
 */
World::World(
    const string &world_name,
    double t_max,
    double delta_n,
    double tau,
    double duo_update_time,
    double duo_update_weight,
    double route_choice_uncertainty,
    int print_mode,
    long long random_seed,
    bool vehicle_log_mode)
    : timestamp(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
      name(world_name),
      t_max(t_max),
      delta_n(delta_n),
      tau(tau),
      duo_update_time(duo_update_time),
      duo_update_weight(duo_update_weight),
      print_mode(print_mode),
      delta_t(tau * delta_n),
      total_timesteps((int)(t_max / (tau * delta_n))),
      timestep_for_route_update((int)(duo_update_time / (tau * delta_n))),
      time(0),
      node_id(0),
      link_id(0),
      vehicle_id(0),
      timestep(0),
      route_choice_uncertainty(route_choice_uncertainty),
      random_seed(random_seed),
      vehicle_log_reserve_size(0),
      vehicle_log_mode(vehicle_log_mode),
      ave_v(0.0),
      ave_vratio(0.0),
      trips_total(0.0),
      trips_completed(0.0),
      rng((std::mt19937::result_type)random_seed),
      flag_initialized(false),
      writer(&std::cout){
}

/**
 * 初始化邻接矩阵
 */
void World::initialize_adj_matrix(){
    if (flag_initialized==false){
        adj_mat.resize(node_id, vector<int>(node_id, 0));
        adj_mat_time.resize(node_id, vector<double>(node_id, 0.0));
        for (auto ln : links){
            int i = ln->start_node->id;
            int j = ln->end_node->id;
            adj_mat[i][j] = 1;
            adj_mat_time[i][j] = ln->length / ln->vmax;
        }

        route_preference.resize(nodes.size());
        for (auto nd : nodes){
            for (auto ln : links){
                route_preference[nd->id][ln] = 0.0;
            }
        }
        flag_initialized = true;
    }
}

/**
 * 更新时间邻接矩阵
 */
void World::update_adj_time_matrix(){
    for (auto ln : links){
        int i = ln->start_node->id;
        int j = ln->end_node->id;
        if (ln->traveltime_real[timestep] != 0.0){
            adj_mat_time[i][j] = ln->traveltime_real[timestep];
        }else{
            adj_mat_time[i][j] = ln->length / ln->vmax;
        }
    }
}

/**
 * 全源最短路径搜索（对每个起点执行Dijkstra算法）
 */
pair<vector<vector<double>>, vector<vector<int>>>
  World::route_search_all(const vector<vector<double>> &adj, double infty) {
    int nsize = (int)adj.size();
    if (std::fabs(infty) < 1e-9) {
        infty = 1e15;
    }

    // 构建邻接表
    vector<vector<pair<int, double>>> adj_list(nsize);
    for (int i = 0; i < nsize; i++) {
        for (int j = 0; j < nsize; j++) {
            if (adj[i][j] > 0.0) {
                adj_list[i].push_back({j, adj[i][j]});
            }
        }
    }

    vector<vector<double>> dist(nsize, vector<double>(nsize, infty));
    vector<vector<int>> next_hop(nsize, vector<int>(nsize, -1));

    // 使用优先队列
    using pdi = pair<double, int>;
    std::priority_queue<pdi, vector<pdi>, std::greater<pdi>> pq;

    // 从每个起点执行Dijkstra算法
    for (int start = 0; start < nsize; start++) {
        vector<bool> visited(nsize, false);
        dist[start][start] = 0.0;
        next_hop[start][start] = start;
        pq.push({0.0, start});

        while (!pq.empty()) {
            auto [d, current] = pq.top();
            pq.pop();

            if (visited[current]) continue;
            visited[current] = true;

            // 遍历邻接节点
            for (const auto& [next, weight] : adj_list[current]) {
                double new_dist = dist[start][current] + weight;
                if (new_dist < dist[start][next]) {
                    dist[start][next] = new_dist;
                    // 更新下一跳
                    next_hop[start][next] = (current == start) ?
                                          next : next_hop[start][current];
                    pq.push({new_dist, next});
                }
            }
        }
    }

    return {dist, next_hop};
}

/**
 * DUO路径选择更新
 */
void World::route_choice_duo(){
    for (auto dest : nodes){
        int k = dest->id;

        auto duo_update_weight_tmp = duo_update_weight;
        if (sum_map_values(route_preference[k]) == 0){
             duo_update_weight_tmp = 1; // 使用确定性最短路径初始化
        }

        // 更新每条链路的偏好值
        for (auto ln : links){
            int i = ln->start_node->id;
            int j = ln->end_node->id;
            if (route_next[i][k] == j){
                route_preference[k][ln] = (1.0 - duo_update_weight) * route_preference[k][ln] + duo_update_weight;
            }else{
                route_preference[k][ln] = (1.0 - duo_update_weight) * route_preference[k][ln];
            }
        }
    }
}

/**
 * 打印场景统计信息
 */
void World::print_scenario_stats(){
    if (print_mode == 1){
        (*writer) << "Scenario statistics:\n";
        (*writer) << "    duration: " << t_max << " s\n";
        (*writer) << "    timesteps: " << total_timesteps << "\n";
        (*writer) << "    nodes: " << nodes.size() << "\n";
        (*writer) << "    links: " << links.size() << "\n";
        (*writer) << "    vehicles: " << (int)vehicles.size() * (int)delta_n << " veh\n";
        (*writer) << "    platoon size: " << delta_n << " veh\n";
        (*writer) << "    platoons: " << vehicles.size() << "\n";
    }
}

/**
 * 打印简单结果
 */
void World::print_simple_results(){
    double n = 0.0;

    for (auto veh : vehicles){
        trips_total += delta_n;
        for (int j = 0; j < (int)veh->log_state.size(); j++){
            if (veh->log_state[j] == vsRUN){
                double v_cur = veh->log_v[j];
                ave_v += (v_cur - ave_v) / (n + 1.0);

                Link *ln_ptr = nullptr;
                if (veh->log_link[j] != -1){
                    ln_ptr = get_link_by_id(veh->log_link[j]);
                }
                double denom_vmax = (ln_ptr) ? ln_ptr->vmax : 1.0;
                double vratio = v_cur / denom_vmax;

                ave_vratio += (vratio - ave_vratio) / (n + 1.0);
                n += 1.0;
            }else if (veh->log_state[j] == vsEND){
                trips_completed += delta_n;
                break;
            }
        }
    }

    (*writer) << "Stats:\n";
    (*writer) << "    Average speed: " << ave_v << "\n";
    (*writer) << "    Average speed ratio: " << ave_vratio << "\n";
    (*writer) << "    Trips completion: "
              << trips_completed << " / " << trips_total << "\n";
}

// -----------------------------------------------------------------------
// 主循环
// -----------------------------------------------------------------------

/**
 * 主仿真循环
 */
void World::main_loop(double duration_t=-1, double until_t=-1){
    int start_ts, end_ts;
    start_ts = timestep;

    if (duration_t < 0 && until_t < 0){
        end_ts = total_timesteps;
    } else if (duration_t >= 0 && until_t < 0){
        end_ts = static_cast<size_t>(floor((duration_t+time)/delta_t)) + 1;
    } else if (duration_t < 0 && until_t >= 0){
        end_ts = static_cast<size_t>(floor(until_t/delta_t)) + 1;
    } else {
        throw std::runtime_error("Cannot specify both `duration_t` and `until_t` parameters for `World.main_loop`");
    }

    if (end_ts > (int)total_timesteps){
        end_ts = total_timesteps;
    }
    if (end_ts <= start_ts){
        return;
    }

    for (timestep = start_ts; timestep < (size_t)end_ts; timestep++){
        time = timestep*delta_t;

        // 链路更新
        for (auto ln : links){
            ln->update();
        }

        // 节点生成和信号更新
        for (auto nd : nodes){
            nd->generate();
            nd->signal_update();
        }

        // 节点转移
        for (auto nd : nodes){
            nd->transfer();
        }

        // 跟驰模型
        int veh_count = 0;
        double ave_speed = 0;
        for (const auto& veh : vehicles_running){
            veh.second->car_follow_newell();

            veh_count++;
            ave_speed = ave_speed*(veh_count-1)/veh_count + veh.second->v/(veh_count);
        }

        // 车辆更新（安全遍历，先复制迭代器）
        for (auto it = vehicles_living.begin(); it != vehicles_living.end(); ) {
            Vehicle* veh = it->second;
            ++it;  // 在update之前移动迭代器
            veh->update();
        }

        // 路径选择更新
        if (timestep_for_route_update > 0 && timestep % timestep_for_route_update == 0){
            update_adj_time_matrix();
            auto res = route_search_all(adj_mat_time, 0.0);
            route_dist = res.first;
            route_next = res.second;
            route_choice_duo();
        }

        // 打印进度
        if (print_mode == 1 && total_timesteps > 0 && timestep % (total_timesteps / 10 == 0 ? 1 : total_timesteps / 10) == 0){
            if (timestep == 0){
                (*writer) <<  "Simulating..." << endl;
                (*writer) <<  std::setw(10) << "time"
                    << "|"<< std::setw(14) <<  "# of vehicles"
                    << "|"<< std::setw(11) << " ave speed" << endl;
            }
            (*writer) << std::setw(8) << std::fixed << std::setprecision(0) << time << " s"
                  << "|" << std::setw(10) << veh_count*delta_n << " veh"
                  << "|" << std::setw(7) << std::fixed << std::setprecision(2) << ave_speed << " m/s"
                  << endl;
        }
    }
}

/**
 * 检查仿真是否继续
 */
bool World::check_simulation_ongoing(){
    if (timestep < total_timesteps){
        return true;
    } else {
        return false;
    }
}

// -----------------------------------------------------------------------
// World 工具方法
// -----------------------------------------------------------------------

Node *World::get_node(const string &node_name){
    for (auto nd : nodes){
        if (nd->name == node_name){
            return nd;
        }
    }
    (*writer) << "Error at function get_node(): `"
              << node_name << "` not found\n";
    throw std::runtime_error("get_node() error");
}

Link *World::get_link(const string &link_name){
    for (auto ln : links){
        if (ln->name == link_name){
            return ln;
        }
    }
    (*writer) << "Error at function get_link(): `"
              << link_name << "` not found\n";
    throw std::runtime_error("get_link() error");
}

Vehicle *World::get_vehicle(const string &vehicle_name){
    for (auto vh : vehicles){
        if (vh->name == vehicle_name){
            return vh;
        }
    }
    (*writer) << "Error at function get_vehicle(): `"
              << vehicle_name << "` not found\n";
    throw std::runtime_error("get_vehicle() error");
}

Link *World::get_link_by_id(const int link_id){
    for (auto ln : links){
        if (ln->id == link_id){
            return ln;
        }
    }
    (*writer) << "Error at function get_link_id(): `"
              << link_id << "` not found\n";
    throw std::runtime_error("get_link_id() error");
}

/**
 * 批量分配预定路径（性能优化：减少 Python-C++ 边界跨越）
 *
 * @param assignments vector of (vehicle_index, route_names) pairs
 */
void World::batch_assign_routes(const vector<pair<int, vector<string>>> &assignments) {
    for (const auto &assignment : assignments) {
        int veh_idx = assignment.first;
        const vector<string> &route_names = assignment.second;

        if (veh_idx >= 0 && veh_idx < (int)vehicles.size()) {
            Vehicle *veh = vehicles[veh_idx];
            veh->predefined_route.clear();
            for (const auto &name : route_names) {
                Link *ln = links_map[name];
                if (ln != nullptr) {
                    veh->predefined_route.push_back(ln);
                }
            }
            veh->route_index = 0;
            veh->use_predefined_route = true;
            if (!veh->predefined_route.empty()) {
                veh->route_next_link = veh->predefined_route[0];
            }
        }
    }
}

// -----------------------------------------------------------------------
// add_demand 函数
// -----------------------------------------------------------------------

/**
 * 添加需求（车辆生成）
 *
 * @param use_predefined_route 如果为true，links_preferred_str将作为预定路径使用
 */
inline void add_demand(
        World *w,
        const string &orig_name,
        const string &dest_name,
        double start_t,
        double end_t,
        double flow,
        vector<string> links_preferred_str = {},
        bool use_predefined_route = false){
    double demand = 0.0;
    for (double t = start_t; t < end_t; t += w->delta_t){
        demand += flow * w->delta_t;
        if (demand > (double)w->delta_n){
            Vehicle *v = new Vehicle(
                w,
                orig_name + "-" + dest_name + "-" + std::to_string(t),
                t,
                orig_name,
                dest_name);

            if (use_predefined_route && !links_preferred_str.empty()) {
                // 预定路径模式
                v->assign_route_by_name(links_preferred_str);
            } else {
                // DUO偏好模式（原有逻辑）
                for (auto ln_str : links_preferred_str){
                    v->links_preferred.push_back(w->links_map[ln_str]);
                }
            }

            demand -= (double)w->delta_n;
        }
    }
}
