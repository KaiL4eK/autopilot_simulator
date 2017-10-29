import sys
sys.path.append('../')

from qfs.sim_map import *

# import pygame
# pygame.init()

import sys

if len(sys.argv) != 2:
	print('Bad args')
	exit(1)

filename = sys.argv[1]
resolution_m_px = 0.03

# def mouse_check(event,x,y,flags,param):
#     if event == cv2.EVENT_MOUSEMOVE:
#         cv2.putText(draw_img,"({}, {})".format(x, y),(0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0))

# cv2.namedWindow('1')
# cv2.setMouseCallback('1',mouse_check)

sim_map = get_map_from_file(filename)
sim_map_img = sim_map.get_image(resolution_m_px)
draw_img = cv2.flip(np.copy(sim_map_img), 0)
# surf = pygame.surfarray.make_surface(sim_map_img)

# screen = pygame.display.set_mode((sim_map_img.shape[1], sim_map_img.shape[0]))

# running = True

# while running:
# 	for event in pygame.event.get():
# 		if event.type == pygame.QUIT:
# 			running = False

# 	screen.blit(surf,(0,0))
# 	pygame.display.update()
# pygame.quit()

while(1):
	cv2.imshow('1', draw_img)
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
