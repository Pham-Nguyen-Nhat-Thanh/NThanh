This script implements a discrete-event simulation of internal container-terminal operations, focusing on collaborative scheduling among Quay Cranes (QC), Trucks, and Yard Cranes (YC). The simulator generates a synthetic vessel-call scenario with both unloading and loading tasks, then runs an event-based schedule where each equipment unit processes one task at a time under precedence constraints (unload flow: QC → Truck → YC; load flow: YC → Truck → QC). The objective is to minimize the overall makespan, i.e., the maximum completion time across all equipment. 
A Deep Double Q-Network (DDQN) agent is integrated to learn adaptive dispatching for trucks. Whenever a truck becomes available, the agent selects one heuristic rule from a fixed action space (e.g., shortest/longest processing or setup/transition time), and the chosen heuristic is used to pick the next feasible truck task. QC and YC follow a default heuristic policy. The agent is trained online during simulation using a makespan-based reward. 


**In Test4.py**

_Inputs (edit in main() via SimInput cfg):_
- num_tasks: total number of container tasks in the scenario.
- num_unload: number of unloading tasks (remaining tasks are loading).
- num_qc, num_yc, num_trucks: number of QCs, YCs, and trucks.
- qc_range, yc_range, truck_range (EquipDistRange): uniform ranges for drawing mean/STD of processing and transition times; per task × per equipment times are sampled from these ranges to model operational uncertainty.
- seed: random seed for reproducible instances. 

_Outputs (returned each episode):_
- M_value (Makespan): final completion time of the scenario.
- completed: number of fully completed tasks.
- Equipment schedules: qc_scheduler, yc_scheduler, truck_scheduler with task counts and completion timelines.
- reward: episode reward used for DDQN training.
- logger: detailed dispatching and timing history; printed at the final episode. 

_How to use:_
- Open main() and modify SimInput cfg to match your terminal scale and timing assumptions.
- Set num_episodes for training length.
- Run python Test4.py. The console prints makespan, completion count, and reward per episode, plus a detailed schedule report at the end. 
