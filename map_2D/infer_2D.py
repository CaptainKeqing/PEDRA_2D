import Bayesian_Hilbert_Maps.BHM.original.sbhm as sbhm
from map_2D.aux_funcs import *
import map_2D.agent2D as agent2D
import time
import map_2D.rrt_BHM as rrt_BHM
import matplotlib.pyplot as plt

valid_starting_points = [(113, 76), (52, 125), (182, 187), (81, 18), (75, 197), (193, 107), (151, 162)]  # X, Y

# Training map
gt = get_ground_truth_array(r'C:\Users\USER\IdeaProjects\PEDRA_CPU\map_2D\environments\filled_simple_floorplan.png')
# plt.imshow(gt, 'Greys_r')
# plt.show()

# Paths
custom_load_dir = 'C:/Users/USER/IdeaProjects/PEDRA_CPU/map_2D/results/weights/drone_2D_4000'
log_dir = 'C:/Users/USER/IdeaProjects/PEDRA_CPU/map_2D/results/inference/infer_log.txt'

# RRT variables
danger_radius = 4
occ_threshold = 0.7

# SBHM variables
gamma = 0.02
cell_res = (12, 12)
min_max = (0, 223, 0, 223)
LIDAR_max_range = 50        # TODO: I'm not sure if changing the max range will affect something.. But i want it to be more of a challenge for this environment

BHM = sbhm.SBHM(gamma=gamma, cell_resolution=cell_res, cell_max_min=min_max)

# agent
drone = agent2D.agent_2D(BHM=BHM, min_max=min_max, LIDAR_pixel_range=LIDAR_max_range, ground_truth_map=gt, starting_pos=valid_starting_points[1],
                         plot_dir='', weights_dir='', custom_load=custom_load_dir)
drone.collect_data()    # need to do 1 fitting of BHM first before can query
current_state = drone.get_state()

# Inference Variables
cum_path_length = 0
minimum_finished_ratio = 0.77

plt.ion()
plt.show()
log_file = open(log_dir, mode='w')
print("******** INFERENCE BEGINS *********")
while True:
    action = drone.network_model.action_selection(current_state)
    drone.network_model.
    print("Action Selected:", action[0])
    print("Coordinate",action_idx_to_coords(action[0], min_max))
    log_file.write("Action Selected: {}".format(action[0]))
    log_file.write("Coordinate: {}".format(action_idx_to_coords(action[0], min_max)))
    # RRT* Algo
    startpos = drone.position
    goalpos = action_idx_to_coords(action[0], min_max)

    G = rrt_BHM.Graph(startpos, goalpos, min_max)
    G = rrt_BHM.RRT_n_star(G, drone.BHM, n_iter=450, radius=5, stepSize=14, crash_radius=5, n_retries_allowed=0)

    if G.success:
        path = rrt_BHM.dijkstra(G)

        path = [(int(elem[0]), int(elem[1])) for elem in path]

        _, path_length = drone.move_by_sequence(path[1:])  # exclude first point
        cum_path_length += path_length

    else:
        path_length = 0

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
        print("Finished ratio:", finished_ratio)
        log_file.write("Finished ratio: {}".format(finished_ratio))

        if finished_ratio > minimum_finished_ratio:
            done = True

        new_state = drone.get_state()

    else:
        new_state = current_state

    if done:
        print("******** EXPLORATION DONE *********")
        print("Path Length:", cum_path_length)
        log_file.write("Path Length: {}".format(path_length))
        print("Finished ratio:", finished_ratio)
        break

    else:
        current_state = new_state





