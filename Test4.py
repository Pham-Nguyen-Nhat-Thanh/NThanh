import numpy as np
import random
from collections import deque
from typing import List, Dict, Tuple, Any, Optional
import tensorflow as tf
import math
from enum import Enum
import heapq
from dataclasses import dataclass

# ==================== ENUM & DATA CLASSES ====================

class EdgeType(Enum):
    BLUE_EQUIP_UNIQUE = "blue_bidirectional"
    GREEN_TRUCK_UNIQUE = "green_bidirectional"
    BLACK_PRECEDENCE = "black_unidirectional"

class Action(Enum):
    STT = 0  # Shortest Task Switching Time
    LTT = 1  # Longest Task Switching Time
    SPT = 2  # Shortest Processing Time
    LPT = 3  # Longest Processing Time
    LME = 4  # Maximum Mapping Entropy

@dataclass
class TaskEquipment:
    processing_time: Dict[int, float]  # key = equip_id
    transition_time: Dict[int, float]

@dataclass
class EquipSets:
    qc_ids: List[int]
    truck_ids: List[int]
    yc_ids: List[int]

class TaskState:
    def __init__(self, task_id, kind):
        self.task_id = task_id
        self.kind = kind
        self.qc_assigned = False
        self.qc_completed = False
        self.yc_assigned = False
        self.yc_completed = False
        self.truck_assigned = False
        self.truck_completed = False
        self.fully_completed = False
        self.assigned_truck = None
        self.assigned_qc = None
        self.assigned_yc = None
        self.start_time = None
        self.completion_time = None
        self.current_stage = "PENDING"  # QC, TRUCK, YC, COMPLETED

@dataclass
class ContainerTaskSpec:
    cid: int
    kind: str  # 'unload' or 'load'
    qc_cands: List[int]
    truck_cands: List[int]
    yc_cands: List[int]
    qc_equipment: TaskEquipment
    yc_equipment: TaskEquipment
    truck_equipment: TaskEquipment

    def __post_init__(self):
        self.state = TaskState(self.cid, self.kind)
        self.qc_heuristic = Action.STT
        self.yc_heuristic = Action.STT
        self.truck_heuristic = Action.STT

@dataclass
class SimpleGraph:
    nodes: List[Tuple[str, int]]
    edges: List[Tuple[int, int, EdgeType]]
    index: Dict[Tuple[str, int], int]
    meta: Dict[str, Any]
    task_node_ids: Dict[int, Dict[str, int]]
    equip_node_ids: Dict[str, Dict[int, int]]

# ===== NEW: cấu hình phân phối cho từng loại thiết bị (μ, σ lấy từ Uniform) =====
@dataclass
class EquipDistRange:
    mu_proc: Tuple[float, float]
    sigma_proc: Tuple[float, float]
    mu_trans: Tuple[float, float]
    sigma_trans: Tuple[float, float]

@dataclass
class SimInput:
    num_tasks: int
    num_unload: int
    num_qc: int
    num_yc: int
    num_trucks: int
    qc_range: EquipDistRange
    yc_range: EquipDistRange
    truck_range: EquipDistRange
    seed: Optional[int] = None  # để tái lập

# ==================== GRAPH MODELING ====================

def build_graph_model(equip: EquipSets, tasks: List[ContainerTaskSpec]) -> SimpleGraph:
    nodes = []
    index = {}
    edges = []
    task_node_ids = {}
    equip_node_ids = {"QC": {}, "TRUCK": {}, "YC": {}}

    for q in equip.qc_ids:
        idx = len(nodes); nodes.append(("QC", q))
        index[("QC", q)] = idx; equip_node_ids["QC"][q] = idx
    for k in equip.truck_ids:
        idx = len(nodes); nodes.append(("TRUCK", k))
        index[("TRUCK", k)] = idx; equip_node_ids["TRUCK"][k] = idx
    for y in equip.yc_ids:
        idx = len(nodes); nodes.append(("YC", y))
        index[("YC", y)] = idx; equip_node_ids["YC"][y] = idx

    for t in tasks:
        idx_qc = len(nodes); nodes.append(("TASK_QC", t.cid)); index[("TASK_QC", t.cid)] = idx_qc
        idx_tr = len(nodes); nodes.append(("TASK_TRUCK", t.cid)); index[("TASK_TRUCK", t.cid)] = idx_tr
        idx_yc = len(nodes); nodes.append(("TASK_YC", t.cid)); index[("TASK_YC", t.cid)] = idx_yc
        task_node_ids[t.cid] = {"qc": idx_qc, "truck": idx_tr, "yc": idx_yc}

        if t.kind == 'unload':
            edges.append((idx_qc, idx_tr, EdgeType.BLACK_PRECEDENCE))
            edges.append((idx_tr, idx_yc, EdgeType.BLACK_PRECEDENCE))
        else:  # load
            edges.append((idx_yc, idx_tr, EdgeType.BLACK_PRECEDENCE))
            edges.append((idx_tr, idx_qc, EdgeType.BLACK_PRECEDENCE))

    for t in tasks:
        idx_qc_task = index[("TASK_QC", t.cid)]
        for q in t.qc_cands:
            v = index[("QC", q)]
            edges.append((idx_qc_task, v, EdgeType.BLUE_EQUIP_UNIQUE))
            edges.append((v, idx_qc_task, EdgeType.BLUE_EQUIP_UNIQUE))
        idx_yc_task = index[("TASK_YC", t.cid)]
        for y in t.yc_cands:
            v = index[("YC", y)]
            edges.append((idx_yc_task, v, EdgeType.BLUE_EQUIP_UNIQUE))
            edges.append((v, idx_yc_task, EdgeType.BLUE_EQUIP_UNIQUE))
    for t in tasks:
        idx_tr_task = index[("TASK_TRUCK", t.cid)]
        for k in t.truck_cands:
            v = index[("TRUCK", k)]
            edges.append((idx_tr_task, v, EdgeType.GREEN_TRUCK_UNIQUE))
            edges.append((v, idx_tr_task, EdgeType.GREEN_TRUCK_UNIQUE))

    meta = {
        "n_qc": len(equip.qc_ids),
        "n_truck": len(equip.truck_ids),
        "n_yc": len(equip.yc_ids),
        "n_containers": len(tasks)
    }

    return SimpleGraph(nodes=nodes, edges=edges, index=index, meta=meta,
                       task_node_ids=task_node_ids, equip_node_ids=equip_node_ids)

# ==================== STATE SPACE ====================

def _neighbors_undirected(G: SimpleGraph, i: int) -> List[int]:
    Ns = set()
    for u, v, _ in G.edges:
        if u == i: Ns.add(v)
        if v == i: Ns.add(u)
    Ns.discard(i)
    return list(Ns)

def mapping_entropy_eq12(G: SimpleGraph, node_idx: int) -> float:
    V = len(G.nodes)
    if V <= 1: return 0.0
    K = len(_neighbors_undirected(G, node_idx))
    if K == 0: return 0.0
    ratio = K / (V - 1.0)
    sum_term = K * math.log(ratio) if ratio > 0 else 0
    return float(-ratio * sum_term)

def build_state_vector(G: SimpleGraph,
                       available_task_nodes: List[int],
                       proc_time: Dict[int, float],
                       trans_time: Dict[int, float],
                       Nmax: int) -> np.ndarray:
    features = []
    chosen = available_task_nodes[:Nmax]
    for node_idx in chosen:
        pt = float(proc_time.get(node_idx, -1.0))
        tt = float(trans_time.get(node_idx, -1.0))
        H  = float(mapping_entropy_eq12(G, node_idx))
        features.extend([pt, tt, H])
    missing = Nmax - len(chosen)
    if missing > 0:
        features.extend([-1.0, -1.0, -1.0] * missing)
    return np.asarray(features, dtype=np.float32)

# ==================== HEURISTICS ====================

def select_task_by_equipment_heuristic(available_tasks, equipment_type, equip_id, action, current_time, G: SimpleGraph = None):
    if not available_tasks: return None, 0.0, 0.0, 0.0
    scores = []
    for task in available_tasks:
        if equipment_type == 'QC':
            proc_time = task.qc_equipment.processing_time[equip_id]
            trans_time = task.qc_equipment.transition_time[equip_id]
        elif equipment_type == 'YC':
            proc_time = task.yc_equipment.processing_time[equip_id]
            trans_time = task.yc_equipment.transition_time[equip_id]
        else:
            proc_time = task.truck_equipment.processing_time[equip_id]
            trans_time = task.truck_equipment.transition_time[equip_id]

        if action == Action.STT: score = trans_time   # minimize
        elif action == Action.LTT: score = trans_time # maximize
        elif action == Action.SPT: score = proc_time  # minimize
        elif action == Action.LPT: score = proc_time  # maximize
        elif action == Action.LME:
            if G is None: score = 0.0
            else:
                node_idx = G.task_node_ids[task.cid]["truck" if equipment_type=='TRUCK' else equipment_type.lower()]
                score = mapping_entropy_eq12(G, node_idx)
        else: score = 0.0
        scores.append((task, score, proc_time, trans_time))
    if action in [Action.STT, Action.SPT]:
        scores.sort(key=lambda x: x[1])            # min
    else:
        scores.sort(key=lambda x: x[1], reverse=True)  # max
    return scores[0] if scores else (None, 0.0, 0.0, 0.0)

# ==================== DDQN ====================

class DDQNAgent:
    def __init__(self, state_size=15, action_size=5, L=100000,
                 epsilon=1.0, epsilon_decay=0.9, epsilon_min=0.01,
                 gamma=0.99, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.L = L
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.replay_buffer = deque(maxlen=4000)
        self.current_q_network = self._build_q_network()
        self.target_q_network  = self._build_q_network()
        self.target_q_network.set_weights(self.current_q_network.get_weights())
        self.train_step_counter = 0
        self.target_update_frequency = 100

    def _build_q_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def get_state(self, G: SimpleGraph, available_task_nodes: List[int],
                  proc_time: Dict[int, float], trans_time: Dict[int, float]) -> np.ndarray:
        return build_state_vector(G, available_task_nodes, proc_time, trans_time, 5)  # Nmax=5

    def select_action(self, state: np.ndarray) -> Action:
        if np.random.rand() <= self.epsilon:
            return random.choice(list(Action))
        q = self.current_q_network.predict(state.reshape(1, -1), verbose=0)[0]
        return list(Action)[int(np.argmax(q))]

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self, batch_size=32):
        if len(self.replay_buffer) < batch_size: return
        minibatch = random.sample(self.replay_buffer, batch_size)
        states      = np.array([e[0] for e in minibatch])
        actions_idx = [e[1].value for e in minibatch]
        rewards     = np.array([e[2] for e in minibatch])
        next_states = np.array([e[3] for e in minibatch])
        dones       = np.array([e[4] for e in minibatch])

        q_curr = self.current_q_network.predict(states, verbose=0)
        q_next = self.target_q_network.predict(next_states, verbose=0)
        max_next = np.max(q_next, axis=1)

        targets = q_curr.copy()
        for i in range(len(minibatch)):
            targets[i, actions_idx[i]] = rewards[i] if dones[i] else rewards[i] + self.gamma * max_next[i]

        self.current_q_network.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_frequency == 0:
            self.target_q_network.set_weights(self.current_q_network.get_weights())

# ==================== SCHEDULERS & EVENTS ====================

class EquipmentScheduler:
    def __init__(self, equip_type, equip_ids):
        self.equip_type = equip_type
        self.equip_ids = equip_ids
        self.available_time = {eid: 0.0 for eid in equip_ids}
        self.current_tasks  = {eid: None for eid in equip_ids}
        self.completion_times = []
        self.task_count = {eid: 0 for eid in equip_ids}

    def get_available_equipment(self, current_time):
        return [eid for eid in self.equip_ids if self.available_time[eid] <= current_time]

    def assign_task(self, equip_id, task, start_time, duration):
        self.current_tasks[equip_id] = task
        end_time = start_time + duration
        self.available_time[equip_id] = end_time
        self.completion_times.append(end_time)
        self.task_count[equip_id] += 1
        return end_time

class SimulationTimeManager:
    def __init__(self):
        self.current_time = 0.0
        self.event_queue = []

    def add_event(self, t, kind, task_id=None, equip_type=None, equip_id=None):
        heapq.heappush(self.event_queue, (t, kind, task_id, equip_type, equip_id))

    def process_next_event(self):
        if not self.event_queue: return None
        ev = heapq.heappop(self.event_queue); self.current_time = ev[0]; return ev

# ==================== TASK SELECTION LOGIC ====================

def get_available_tasks_for_equipment(tasks, equipment_type):
    available = []
    for task in tasks:
        if equipment_type == 'QC':
            if task.kind == 'unload' and not task.state.qc_assigned and not task.state.qc_completed:
                available.append(task)
            elif task.kind == 'load' and task.state.truck_completed and not task.state.qc_assigned and not task.state.qc_completed:
                available.append(task)
        elif equipment_type == 'YC':
            if task.kind == 'load' and not task.state.yc_assigned and not task.state.yc_completed:
                available.append(task)
            elif task.kind == 'unload' and task.state.truck_completed and not task.state.yc_assigned and not task.state.yc_completed:
                available.append(task)
        elif equipment_type == 'TRUCK':
            if task.kind == 'unload' and task.state.qc_completed and not task.state.truck_assigned and not task.state.truck_completed:
                available.append(task)
            elif task.kind == 'load' and task.state.yc_completed and not task.state.truck_assigned and not task.state.truck_completed:
                available.append(task)
    return available

# ==================== OBJECTIVE & REWARD ====================

def compute_objective_M(qc_scheduler, truck_scheduler, yc_scheduler):
    qc_max = max(qc_scheduler.completion_times) if qc_scheduler.completion_times else 0
    tr_max = max(truck_scheduler.completion_times) if truck_scheduler.completion_times else 0
    yc_max = max(yc_scheduler.completion_times) if yc_scheduler.completion_times else 0
    return max(qc_max, tr_max, yc_max)

def reward_eq14(task_completed: bool, M_value: float, L: float = 1.0) -> float:
    if not task_completed or M_value <= 0.0: return 0.0
    return L * (1.0 / M_value)

# ==================== LOGGING ====================

class TaskLogger:
    def __init__(self):
        self.task_history = []
        self.equipment_utilization = {'QC': {}, 'TRUCK': {}, 'YC': {}}
    def log_task_start(self, task_id, equipment_type, equipment_id, start_time, action, heuristic):
        self.task_history.append({
            'task_id': task_id, 'equipment_type': equipment_type, 'equipment_id': equipment_id,
            'start_time': start_time, 'action': action, 'heuristic': heuristic,
            'completion_time': None, 'duration': None
        })
    def log_task_completion(self, task_id, equipment_type, completion_time):
        for log in self.task_history:
            if log['task_id']==task_id and log['equipment_type']==equipment_type and log['completion_time'] is None:
                log['completion_time'] = completion_time
                log['duration'] = completion_time - log['start_time']; break
    def log_equipment_usage(self, equipment_type, equipment_id, duration):
        d = self.equipment_utilization[equipment_type]
        d[equipment_id] = d.get(equipment_id, 0.0) + duration
    def print_detailed_analysis(self, tasks, total_duration, qc_scheduler, yc_scheduler, truck_scheduler):
        print("\n"+"="*80); print("DETAILED SCHEDULING ANALYSIS"); print("="*80)
        print(f"\nTOTAL MAKESPAN (M): {total_duration:.2f} units")
        print("\nTASK EXECUTION DETAILS:"); print("-"*120)
        print(f"{'Task ID':<8} {'Type':<10} {'Truck':<8} {'QC':<5} {'YC':<5} {'Start':<8} {'Complete':<10} {'Duration':<10} {'Heuristic':<10}")
        print("-"*120)
        for task in sorted(tasks, key=lambda x: x.state.start_time if x.state.start_time is not None else float('inf')):
            start = task.state.start_time or 0.0
            comp  = task.state.completion_time or 0.0
            dur   = comp - start if comp>0 else 0.0
            truck_str = str(task.state.assigned_truck) if task.state.assigned_truck is not None else 'N/A'
            qc_str    = str(task.state.assigned_qc)    if task.state.assigned_qc    is not None else 'N/A'
            yc_str    = str(task.state.assigned_yc)    if task.state.assigned_yc    is not None else 'N/A'
            heur      = getattr(task, 'truck_heuristic', Action.STT)
            print(f"{task.cid:<8} {task.kind:<10} {truck_str:<8} {qc_str:<5} {yc_str:<5} {start:<8.2f} {comp:<10.2f} {dur:<10.2f} {heur.name:<10}")

# ==================== FLEXIBLE DATA GENERATION (the new part) ====================

def _abs_normal(mu: float, sigma: float) -> float:
    v = abs(np.random.normal(mu, sigma))
    return v

# def _gen_time(proc_mu, proc_sigma, trans_mu, trans_sigma):
#     proc = max(_abs_normal(proc_mu, proc_sigma), 0.1)  # sàn 0.1
#     trans = max(_abs_normal(trans_mu, trans_sigma), 0.0)
#     return proc, trans

# def _sample_equip_params(n: int, rng: EquipDistRange) -> List[Dict[str, float]]:
#     """Mỗi thiết bị i có (μ,σ) riêng; tất cả rút từ Uniform trong khoảng cấu hình."""
#     params = []
#     for _ in range(n):
#         mu_p  = np.random.uniform(*rng.mu_proc)
#         sg_p  = np.random.uniform(*rng.sigma_proc)
#         mu_t  = np.random.uniform(*rng.mu_trans)
#         sg_t  = np.random.uniform(*rng.sigma_trans)
#         params.append(dict(mu_proc=mu_p, sigma_proc=sg_p, mu_trans=mu_t, sigma_trans=sg_t))
#     return params
def _draw_params_from_uniform(rng):  # rng: EquipDistRange
    mu_p  = np.random.uniform(*rng.mu_proc)
    sg_p  = np.random.uniform(*rng.sigma_proc)
    mu_t  = np.random.uniform(*rng.mu_trans)
    sg_t  = np.random.uniform(*rng.sigma_trans)
    return mu_p, sg_p, mu_t, sg_t


def build_random_instance(cfg: SimInput) -> Tuple[List[ContainerTaskSpec], EquipSets]:
    if cfg.seed is not None:
        np.random.seed(cfg.seed); random.seed(cfg.seed)

    qc_ids    = list(range(cfg.num_qc))
    yc_ids    = list(range(cfg.num_yc))
    truck_ids = list(range(cfg.num_trucks))
    equip = EquipSets(qc_ids=qc_ids, truck_ids=truck_ids, yc_ids=yc_ids)

    assert 0 <= cfg.num_unload <= cfg.num_tasks
    num_load = cfg.num_tasks - cfg.num_unload
    kinds = (['unload'] * cfg.num_unload) + (['load'] * num_load)

    tasks: List[ContainerTaskSpec] = []
    for cid, kind in enumerate(kinds):
        # QC times (per task × per QC)
        qc_proc, qc_trans = {}, {}
        for q in qc_ids:
            mu_p, sg_p, mu_t, sg_t = _draw_params_from_uniform(cfg.qc_range)
            pr = max(abs(np.random.normal(mu_p, sg_p)), 0.1)
            tr = max(abs(np.random.normal(mu_t, sg_t)), 0.0)
            qc_proc[q], qc_trans[q] = pr, tr

        # YC times (per task × per YC)
        yc_proc, yc_trans = {}, {}
        for y in yc_ids:
            mu_p, sg_p, mu_t, sg_t = _draw_params_from_uniform(cfg.yc_range)
            pr = max(abs(np.random.normal(mu_p, sg_p)), 0.1)
            tr = max(abs(np.random.normal(mu_t, sg_t)), 0.0)
            yc_proc[y], yc_trans[y] = pr, tr

        # TRUCK times (per task × per Truck)
        tr_proc, tr_trans = {}, {}
        for k in truck_ids:
            mu_p, sg_p, mu_t, sg_t = _draw_params_from_uniform(cfg.truck_range)
            pr = max(abs(np.random.normal(mu_p, sg_p)), 0.1)
            tr = max(abs(np.random.normal(mu_t, sg_t)), 0.0)
            tr_proc[k], tr_trans[k] = pr, tr

        task = ContainerTaskSpec(
            cid=cid, kind=kind,
            qc_cands=qc_ids.copy(), yc_cands=yc_ids.copy(), truck_cands=truck_ids.copy(),
            qc_equipment=TaskEquipment(processing_time=qc_proc, transition_time=qc_trans),
            yc_equipment=TaskEquipment(processing_time=yc_proc, transition_time=yc_trans),
            truck_equipment=TaskEquipment(processing_time=tr_proc, transition_time=tr_trans)
        )
        tasks.append(task)

    return tasks, equip

# ==================== SIMULATION (accepts tasks & equip) ====================

def run_simulation_with_ddqn(tasks: List[ContainerTaskSpec], equip: EquipSets):
    G = build_graph_model(equip, tasks)
    agent = DDQNAgent(state_size=15, action_size=5, L=1.0)

    qc_scheduler    = EquipmentScheduler('QC', equip.qc_ids)
    yc_scheduler    = EquipmentScheduler('YC', equip.yc_ids)
    truck_scheduler = EquipmentScheduler('TRUCK', equip.truck_ids)

    time_manager = SimulationTimeManager()
    logger = TaskLogger()
    pending_truck: Dict[Tuple[int, int], Tuple[np.ndarray, int]] = {}

    # seed events
    for q in equip.qc_ids:    time_manager.add_event(0.0, 'QC_AVAILABLE',    None, 'QC', q)
    for y in equip.yc_ids:    time_manager.add_event(0.0, 'YC_AVAILABLE',    None, 'YC', y)
    for k in equip.truck_ids: time_manager.add_event(0.0, 'TRUCK_AVAILABLE', None, 'TRUCK', k)

    steps, max_steps = 0, 200000

    while steps < max_steps:
        ev = time_manager.process_next_event()
        if ev is None: break
        current_time, ev_type, task_id, equip_type, equip_id = ev

        if ev_type in ['QC_AVAILABLE','YC_AVAILABLE','TRUCK_AVAILABLE']:
            if ev_type == 'QC_AVAILABLE':
                scheduler, et = qc_scheduler, 'QC'
            elif ev_type == 'YC_AVAILABLE':
                scheduler, et = yc_scheduler, 'YC'
            else:
                scheduler, et = truck_scheduler, 'TRUCK'

            available_tasks = get_available_tasks_for_equipment(tasks, et)
            available_equip  = scheduler.get_available_equipment(current_time)

            if available_equip and available_tasks:
                for eid in available_equip:
                    if not available_tasks: break

                    if et == 'TRUCK':
                        # build state theo node_idx
                        ptd, ttd, nodes = {}, {}, []
                        for t in available_tasks:
                            node_idx = G.task_node_ids[t.cid]["truck"]
                            nodes.append(node_idx)
                            ptd[node_idx] = t.truck_equipment.processing_time[eid]
                            ttd[node_idx] = t.truck_equipment.transition_time[eid]
                        state = agent.get_state(G, nodes, ptd, ttd)

                        action = agent.select_action(state)
                        selected_task, _, ptime, ttime = select_task_by_equipment_heuristic(
                            available_tasks, et, eid, action, current_time, G=G)

                        if selected_task:
                            selected_task.truck_heuristic = action
                            end_time = scheduler.assign_task(eid, selected_task, current_time, ptime + ttime)
                            selected_task.state.truck_assigned = True
                            selected_task.state.assigned_truck = eid
                            selected_task.state.current_stage  = "TRUCK"

                            if selected_task.kind == 'unload':
                                time_manager.add_event(end_time, 'TRUCK_COMPLETED_UNLOAD', selected_task.cid, 'TRUCK', eid)
                            else:
                                time_manager.add_event(end_time, 'TRUCK_COMPLETED_LOAD',   selected_task.cid, 'TRUCK', eid)

                            logger.log_task_start(selected_task.cid, et, eid, current_time, "ASSIGN", action)
                            logger.log_equipment_usage(et, eid, ptime + ttime)
                            pending_truck[(selected_task.cid, eid)] = (state.copy(), action.value)
                            available_tasks = [t for t in available_tasks if t.cid != selected_task.cid]

                    else:
                        heuristic = Action.STT
                        selected_task, _, ptime, ttime = select_task_by_equipment_heuristic(
                            available_tasks, et, eid, heuristic, current_time, G=G)
                        if selected_task:
                            end_time = scheduler.assign_task(eid, selected_task, current_time, ptime + ttime)
                            if et == 'QC':
                                selected_task.state.qc_assigned = True
                                selected_task.state.assigned_qc = eid
                                selected_task.state.current_stage = "QC"
                                if selected_task.state.start_time is None: selected_task.state.start_time = current_time
                                time_manager.add_event(end_time, 'QC_COMPLETED', selected_task.cid, 'QC', eid)
                            else:
                                selected_task.state.yc_assigned = True
                                selected_task.state.assigned_yc = eid
                                selected_task.state.current_stage = "YC"
                                if selected_task.state.start_time is None: selected_task.state.start_time = current_time
                                time_manager.add_event(end_time, 'YC_COMPLETED', selected_task.cid, 'YC', eid)

                            logger.log_task_start(selected_task.cid, et, eid, current_time, "ASSIGN", heuristic)
                            logger.log_equipment_usage(et, eid, ptime + ttime)
                            available_tasks = [t for t in available_tasks if t.cid != selected_task.cid]

        elif ev_type == 'QC_COMPLETED':
            t = tasks[task_id]; t.state.qc_completed = True
            logger.log_task_completion(task_id, 'QC', current_time)
            if t.kind == 'load':
                t.state.fully_completed = True; t.state.completion_time = current_time
            else:
                t.state.current_stage = "TRUCK"
                time_manager.add_event(current_time, 'TRUCK_AVAILABLE', None, 'TRUCK', None)
            time_manager.add_event(current_time, 'QC_AVAILABLE', None, 'QC', equip_id)

        elif ev_type == 'YC_COMPLETED':
            t = tasks[task_id]; t.state.yc_completed = True
            logger.log_task_completion(task_id, 'YC', current_time)
            if t.kind == 'unload':
                t.state.fully_completed = True; t.state.completion_time = current_time
            else:
                t.state.current_stage = "TRUCK"
                time_manager.add_event(current_time, 'TRUCK_AVAILABLE', None, 'TRUCK', None)
            time_manager.add_event(current_time, 'YC_AVAILABLE', None, 'YC', equip_id)

        elif ev_type in ['TRUCK_COMPLETED_UNLOAD','TRUCK_COMPLETED_LOAD']:
            t = tasks[task_id]; t.state.truck_completed = True
            logger.log_task_completion(task_id, 'TRUCK', current_time)
            if ev_type == 'TRUCK_COMPLETED_UNLOAD':
                t.state.current_stage = "YC"
                time_manager.add_event(current_time, 'YC_AVAILABLE', None, 'YC', None)
            else:
                t.state.current_stage = "QC"
                time_manager.add_event(current_time, 'QC_AVAILABLE', None, 'QC', None)
            time_manager.add_event(current_time, 'TRUCK_AVAILABLE', None, 'TRUCK', equip_id)

            # học DDQN cho truck
            key = (task_id, equip_id)
            if key in pending_truck:
                prev_state, a_idx = pending_truck.pop(key)
                M_now = compute_objective_M(qc_scheduler, truck_scheduler, yc_scheduler)
                r = reward_eq14(True, M_now, agent.L)

                next_tasks = get_available_tasks_for_equipment(tasks, 'TRUCK')
                ptd, ttd, nodes = {}, {}, []
                for tt in next_tasks:
                    node_idx = G.task_node_ids[tt.cid]["truck"]
                    nodes.append(node_idx)
                    ptd[node_idx] = tt.truck_equipment.processing_time.get(equip_id, 0.0)
                    ttd[node_idx] = tt.truck_equipment.transition_time.get(equip_id, 0.0)
                next_state = agent.get_state(G, nodes, ptd, ttd)
                done = all(x.state.fully_completed for x in tasks)
                agent.store_experience(prev_state, Action(a_idx), r, next_state, done)
                agent.train()

        steps += 1

    M_value = compute_objective_M(qc_scheduler, truck_scheduler, yc_scheduler)
    completed = sum(1 for t in tasks if t.state.fully_completed)
    reward = reward_eq14(completed > 0, M_value, 1.0)
    return M_value, completed, qc_scheduler, yc_scheduler, truck_scheduler, logger, reward

# ==================== MAIN (example) ====================

def main():
    print("=== FLEXIBLE CONTAINER TERMINAL SIMULATION WITH DDQN ===")

    # Ví dụ cấu hình đầu vào (bạn thay số ở đây cho case của mình)
    cfg = SimInput(
        num_tasks=4,
        num_unload=3,              # số unloading, còn lại = loading
        num_qc=2,
        num_yc=1,
        num_trucks=2,
        qc_range=EquipDistRange(   # Uniform ranges cho QC
            mu_proc=(215, 245), sigma_proc=(70, 100),
            mu_trans=(180, 210), sigma_trans=(60, 80)
        ),
        yc_range=EquipDistRange(   # Uniform ranges cho YC
            mu_proc=(100, 150), sigma_proc=(30, 55),
            mu_trans=(70, 120), sigma_trans=(20, 40)
        ),
        truck_range=EquipDistRange( # Uniform ranges cho Truck
            mu_proc=(240, 280), sigma_proc=(50, 100),
            mu_trans=(200, 240), sigma_trans=(30, 70)
        ),
        seed=2811
    )

    # Tạo instance theo input linh hoạt
    tasks, equip = build_random_instance(cfg)

    num_episodes = 200
    total_rewards = []

    for ep in range(num_episodes):
        print(f"\n--- Episode {ep+1} ---")
        # reset trạng thái task trước mỗi episode
        for t in tasks: t.state = TaskState(t.cid, t.kind)

        M, done_count, qc_sch, yc_sch, tr_sch, logger, rew = run_simulation_with_ddqn(tasks, equip)
        total_rewards.append(rew)
        print(f"Episode {ep+1}: Makespan = {M:.2f}, Completed = {done_count}/{len(tasks)}, Reward = {rew:.4f}")

        if ep == num_episodes - 1:
            logger.print_detailed_analysis(tasks, M, qc_sch, yc_sch, tr_sch)

    print(f"\n=== TRAINING COMPLETED ===")
    print(f"Average Reward: {np.mean(total_rewards):.4f}")

if __name__ == "__main__":
    main()
