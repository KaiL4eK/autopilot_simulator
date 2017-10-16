from simulate_robot import *
import time

simulation_seconds = 30.0
map_filename 	= 'two_obstacles.pmap'
filename 		= 'maze.pmap'
sim_map = get_map_from_file(map_filename)

check_count = 20
result_real_time = 0

for i in range(check_count):

	sim = SimManager(bot=Robot(x=2, y=10),
                     target=CircleTarget(x=36, y=2),
	                 map_data=sim_map)

	start_time = time.time()

	while sim.t < simulation_seconds:
	    sim.sample_step([0, 0, 1])

	end_time = time.time()

	result_real_time += (end_time - start_time)

result_real_time /= check_count

print('Simulation time:', simulation_seconds, 's')
print('Real time:', result_real_time, 's')
print('Time rate:', simulation_seconds / result_real_time)
