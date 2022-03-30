from map_2D.aux_funcs import *
import matplotlib.pyplot as plt
import map_2D.rrt_BHM as rrt_BHM
import map_2D.agent2D as agent2D
import Bayesian_Hilbert_Maps.BHM.original.sbhm as sbhm
import time
gt = get_ground_truth_array(r'C:\Users\USER\IdeaProjects\PEDRA_CPU\map_2D\environments\filled_simple_floorplan.png')

z = np.zeros((2, 2), dtype=np.float32)
z[1, 1] = 1
print(z)
z[0, 0] = 0.05
print(z)
z *= 0.9
print(z)

# plt.imshow(gt, 'Greys_r')
# # plt.scatter(neighbours[:, 0], neighbours[:, 1], cmap='jet')
# # print([gt[p[1], p[0]] for p in neighbours])
# plt.show()
gamma = 0.02
cell_res = (12, 12)
min_max = (0, 223, 0, 223)
LIDAR_max_range = 200

BHM = sbhm.SBHM(gamma=gamma, cell_resolution=cell_res, cell_max_min=min_max)
valid_starting_points = [(50, 25), (68, 75), (100, 75), (35, 162), (69, 170), (83, 201), (164, 165), (153, 75), (173, 32), (200, 71)]
drone = agent2D.agent_2D(BHM=BHM, min_max=min_max, LIDAR_pixel_range=LIDAR_max_range, ground_truth_map=gt, starting_pos=valid_starting_points[0],
                         plot_dir='plot_dir', weights_dir='weights_dir')
drone.collect_data()
for point in valid_starting_points[1:]:
    drone.position = point
    drone.collect_data()

# pred = drone.BHM.predict_proba(points)[:, 1]
# print(pred)
# drone.show_model()
startpos = (18, 193)

goalpos = (200, 100)
G = rrt_BHM.Graph(startpos, goalpos, (0, 223, 0, 223))
s=time.time()
G = rrt_BHM.RRT_n_star_np_arr(G, np.reshape(drone.BHM.predict_proba(drone.qX)[:, 1], (224, 224)), 1000, 5, 14, 5, 0)
print('time tp arr', time.time()-s)
# s=time.time()
# G = rrt_BHM.RRT_n_star(G, drone.BHM, 1000, 5, 14, 5, 0)
# print('time to bhm', time.time()-s)
if G.success:
    path = rrt_BHM.dijkstra(G)
    print(path)
else:
    path = None
rrt_BHM.plot(G, drone.BHM, path)

# pss = [(1, 2), (3, 4)]
# print(pss[1:])
