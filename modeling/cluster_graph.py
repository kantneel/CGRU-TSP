import numpy as np 
import math

def cluster_graph(vertices, max_coord, n_clusters, max_pop, p_rad, p_arc):
	assignments = []
	p_coords = []
	r_low = p_rad * max_coord / 2
	for i in range(n_clusters):
		for _ in range(max_pop):
			assignments.append(i)

	assign = np.array(assignments)
	np.random.shuffle(assign)
	final_assign = assign[:vertices]


	for i in range(vertices):
		r = np.random.uniform(low=r_low, high=max_coord / 2)
		theta = 360 * (final_assign[i] + np.random.uniform(low=0, high=p_arc))  / n_clusters
		p_coords.append([r, theta])

	offset = np.random.uniform(low=0, high=1) * 360 / n_clusters
	p_coords = [[c[0], (c[1] + offset) % 360] for c in p_coords]
	eu_coords = [polar_to_euclid(c) for c in p_coords]

	eu_coords.sort(key=lambda x: distance((0, 0), x))

	edges = np.zeros((num_vertices, num_vertices))
	for i in range(num_vertices):
		for j in range(num_vertices):
			if i == j:
				continue
			edges[i, j] = distance(coords[i], coords[j])

	return edges


def polar_to_euclid(polar):
	r = polar[0]
	theta = math.radians(polar[1])

	return [r * math.cos(theta), r * math.sin(theta)]

