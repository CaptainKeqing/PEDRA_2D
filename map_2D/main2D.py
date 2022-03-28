# FROM THIS POINT OUT, LET COORD BE DESCRIBED AS X, Y.
# To interpret model output, since he gives in terms of 224x224 action space, the first index is actually row (which is Y)
# and the second index is width (which is X)
import Bayesian_Hilbert_Maps.BHM.original.sbhm as sbhm
from map_2D.aux_funcs import *
import map_2D.agent2D as agent2D
import time
import map_2D.rrt_BHM as rrt_BHM
valid_starting_points = [(113, 76), (52, 125), (182, 21), (81, 14), (75, 210), (202, 104), (151, 162)]  # X, Y
import matplotlib.pyplot as plt

# Training map
gt = get_ground_truth_array(r'C:\Users\USER\IdeaProjects\PEDRA_CPU\map_2D\environments\filled_simple_floorplan.png')

# Paths
plot_dir = 'C:/Users/USER/IdeaProjects/PEDRA_CPU/map_2D/results/stats'
weights_dir = 'C:/Users/USER/IdeaProjects/PEDRA_CPU/map_2D/results/weights'
log_dir = 'C:/Users/USER/IdeaProjects/PEDRA_CPU/map_2D/results/log'

# Initialise variables
iter = 0
max_iters = 10000
save_interval = max_iters // 5
level = 0   # if implementing switching starting positions
episode = 0  # how many times drone completed exploration
moves_taken = 0
epsilon_saturation = 10000
epsilon_model = 'exponential'
epsilon = 0  # start with drone always taking random actions
cum_return = 0
discount_factor = 0.8
Q_clip = True   # clips TD error to -1, 1
learning_rate = 2e-5

consecutive_fails = 0
max_consecutive_fails = 5  # for debugging purposes

# RRT variables
danger_radius = 5
occ_threshold = 0.65

# SBHM variables
gamma = 0.02
cell_res = (12, 12)
min_max = (0, 223, 0, 223)
LIDAR_max_range = 50        # TODO: I'm not sure if changing the max range will affect something.. But i want it to be more of a challenge for this environment

BHM = sbhm.SBHM(gamma=gamma, cell_resolution=cell_res, cell_max_min=min_max)

# agent
drone = agent2D.agent_2D(BHM=BHM, min_max=min_max, LIDAR_pixel_range=LIDAR_max_range, ground_truth_map=gt, starting_pos=valid_starting_points[0],
                         plot_dir=plot_dir, weights_dir=weights_dir)
drone.collect_data()    # need to do 1 fitting of BHM first before can query
current_state = drone.get_state()

# plt.ion()
# plt.show()
print("******** SIMULATION BEGINS *********")
# TRAINING LOOP
while True:
    start_time = time.time()

    action, action_type, epsilon = policy_FCQN(epsilon, current_state,
                                               iter, epsilon_saturation, 'exponential', drone)

    # RRT* algo
    startpos = drone.position
    goalpos = action_idx_to_coords(action[0], min_max)

    valid_goal = True

    surroundings = bloom(goalpos, danger_radius, resolution_per_quadrant=16)
    pred_occupancies = drone.BHM.predict_proba(surroundings)[:, 1]
    goal_close_to_obstacle = any(occ_val > occ_threshold for occ_val in pred_occupancies)

    pred_goal = drone.BHM.predict_proba(np.array([goalpos]))[0][1]
    goal_in_unknown_space = 0.4 < pred_goal < 0.65  # roughly, if my probability of being occupied is around 0.5 +- 0.1, means im unsure, which is dangerous

    if pred_goal > occ_threshold or goal_close_to_obstacle:  # point selected is in obstacle / too close
        path = None
        path_length = 0
        safe_travel = None
    else:
        G = rrt_BHM.Graph(startpos, goalpos, min_max)
        G = rrt_BHM.RRT_n_star(G, drone.BHM, n_iter=500, radius=5,      # RRT Params must be modified based on the environment, but this is not an issue of the agent
                               stepSize=12, crash_radius=5, n_retries_allowed=0)
        if G.success:
            path = rrt_BHM.dijkstra(G)
            # print('start', drone.position)
            # print('goal', goalpos)
            # print('action', action[0])
            # rrt_BHM.plot(G, drone.BHM, path)
            path = [(int(elem[0]), int(elem[1])) for elem in path]

            safe_travel, path_length = drone.move_by_sequence(path[1:])  # exclude first point
            if path_length == 0:
                consecutive_fails += 1
                if consecutive_fails == max_consecutive_fails:
                    print("DRONE STUCKKKK")
                    print('drone_pos:', drone.position)
                    print('goal_pos:', goalpos)
                    rrt_BHM.plot(G, drone.BHM, None)
                    consecutive_fails = 0
            else:
                consecutive_fails = 0
            moves_taken += 1
        else:

            path = None
            path_length = 0
            safe_travel = None

    # In very rare cases (happened once after 1500 iters), drone will actually get stuck in a wall. Crash checking.
    # If clip into wall, just do a hard reset and dont do any training for that move.
    # print('drone position:', drone.position)
    neighbours = neighbours_including_center(drone.position)    # radius of 1 only, crash shouldn't happen normally

    if neighbours is None or any(gt[p[1], p[0]] == 0 for p in neighbours):   # drone position is x, y. However to index a 2D array, index by row (y) then col (x)
        print("CRASH OCCURED")
        drone.reset(fresh_BHM=sbhm.SBHM(gamma=gamma, cell_resolution=cell_res,
                                        cell_max_min=min_max),
                    starting_pos=valid_starting_points[0])
        current_state = drone.get_state()
        # don't +1 to episode, treat as same episode and reset move and return
        moves_taken = 0
        cum_return = 0
        continue  # skip rest of the iteration

    reward = drone.reward_gen(path_length, goal_in_unknown_space=goal_in_unknown_space, safe_travel=safe_travel)

    new_state = drone.get_state()

    # check for completeness, only if moved
    done = False
    if path_length != 0:
        free_mask = drone.get_free_mask()
        correct = np.logical_and(gt, free_mask)
        plt.imshow(correct, cmap='Greys_r')
        # plt.scatter(drone.position[0], drone.position[1], cmap='jet')
        plt.draw()
        plt.pause(0.001)
        # drone.show_model()
        finished_ratio = np.sum(correct) / np.sum(gt)
        # print("Finished ratio:", finished_ratio)

        if finished_ratio > 0.80:
            done = True
            reward += 1

    # TRAINING DONE HERE
    cum_return = cum_return + reward

    data_tuple = (current_state, action, new_state, reward)

    _, Q_target, err = get_err_FCQN(data_tuple, drone, discount_factor, Q_clip)

    drone.network_model.train_n(current_state, action, Q_target, 1, learning_rate, epsilon, iter)
    # ------------------

    time_exec = time.time() - start_time

    s_log = 'drone_2D - Level {:>2d} - Iter: {:>5d}/{:<4d} Action: {}-{:>5s} Eps: {:<1.4f} lr: {:>1.5f} Ret = {:<+6.4f} t={:<1.3f} Moves: {:<2} Steps: {:<3} Reward: {:<+1.4f}  '.format(
        level,
        iter,
        episode,
        action,
        action_type,
        epsilon,
        learning_rate,
        cum_return,
        time_exec,
        moves_taken,
        len(drone.previous_positions),
        reward)

    print(s_log)    # TODO: ALSO PRINT TO LOG FILE NEXT TIME

    # IN VERY RARE CASES
    if done:
        drone.network_model.log_to_tensorboard(tag='Return', group='drone_2D',
                                                           value=cum_return,
                                                           index=episode)
        drone.network_model.log_to_tensorboard(tag='Moves (valid goalpoints)', group='drone_2D',
                                                       value=moves_taken,
                                                       index=episode)
        drone.network_model.log_to_tensorboard(tag='Steps (waypoints)', group='drone_2D',
                                               value=len(drone.previous_positions),
                                               index=episode)

        drone.reset(fresh_BHM=sbhm.SBHM(gamma=gamma, cell_resolution=cell_res,
                                        cell_max_min=min_max),
                    starting_pos=valid_starting_points[0])

        current_state = drone.get_state()

        # drone.show_model()
        episode += 1
        moves_taken = 0
        cum_return = 0

    else:
        current_state = new_state

    iter += 1
    if iter % save_interval == 0:
        drone.network_model.save_network(str(iter))
    if iter == max_iters:
        print("TRAINING DONE")
        break
