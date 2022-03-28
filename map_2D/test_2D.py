from map_2D.aux_funcs import *
import matplotlib.pyplot as plt
import map_2D.rrt_BHM as rrt_BHM
import map_2D.agent2D as agent2D
import Bayesian_Hilbert_Maps.BHM.original.sbhm as sbhm
gt = get_ground_truth_array(r'C:\Users\USER\IdeaProjects\PEDRA_CPU\map_2D\environments\filled_simple_floorplan.png')


plt.imshow(gt, 'Greys_r')
# plt.scatter(neighbours[:, 0], neighbours[:, 1], cmap='jet')
# print([gt[p[1], p[0]] for p in neighbours])
plt.show()
gamma = 0.02
cell_res = (12, 12)
min_max = (0, 223, 0, 223)
LIDAR_max_range = 200

BHM = sbhm.SBHM(gamma=gamma, cell_resolution=cell_res, cell_max_min=min_max)
valid_starting_points = [(15, 200), (15, 100), (80, 150), (150, 50), (180, 175)]
drone = agent2D.agent_2D(BHM=BHM, min_max=min_max, LIDAR_pixel_range=LIDAR_max_range, ground_truth_map=gt, starting_pos=valid_starting_points[0],
                         plot_dir='plot_dir', weights_dir='weights_dir')
drone.collect_data()
for point in valid_starting_points[1:]:
    drone.position = point
    drone.collect_data()

# pred = drone.BHM.predict_proba(points)[:, 1]
# print(pred)
drone.show_model()
startpos = (50, 175)

goalpos = (200, 200)
G = rrt_BHM.Graph(startpos, goalpos, (0, 223, 0, 223))
G = rrt_BHM.RRT_n_star(G, drone.BHM, 1000, 5, 12, 5, 0)
if G.success:
    path = rrt_BHM.dijkstra(G)
else:
    path = None
rrt_BHM.plot(G, drone.BHM, path)

pss = [(1, 2), (3, 4)]
print(pss[1:])