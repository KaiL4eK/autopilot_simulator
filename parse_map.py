from __future__ import print_function

import xml.etree.ElementTree as etree

tree = etree.parse('two_obstacles.pmap')
root = tree.getroot()    

print(root.tag, len(root))

for child in root:
	# print(child.tag)

	if child.tag == 'size':
		map_size = (int(child.attrib['width']), int(child.attrib['height']))

		print(map_size)

	if child.tag == 'obstacles':
		for obstacle in child:

			if obstacle.tag == 'rectangle':
				ul = (int(obstacle.attrib['ul_x']), int(obstacle.attrib['ul_y']))
				obst_size = (int(obstacle.attrib['width']), int(obstacle.attrib['height']))

				print(ul, obst_size)
