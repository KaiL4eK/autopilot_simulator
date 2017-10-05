from simulate_robot import *
import time

simulation_seconds = 15.0
map_filename = 'two_obstacles.pmap'
sim_map = get_map_from_file(map_filename)

dt = 1/1000 # 1000 Hz

check_count = 20
result_real_time = 0

for i in range(check_count):

	sim = SimManager(dt=dt,
					 bot=Robot(x=3, y=7.5, theta=0),
	                 target=CircleTarget(x=18, y=5),
	                 map_data=sim_map)

	start_time = time.time()

	while sim.t < simulation_seconds:
	    sim.sample_step([0, 0, 1], False)

	end_time = time.time()

	result_real_time += (end_time - start_time)

result_real_time /= check_count

print('Simulation time:', simulation_seconds, 's')
print('Real time:', result_real_time, 's')
print('Time rate:', simulation_seconds / result_real_time)
