import numpy as np 
import math
import heapq
import itertools
from pprint import pprint
from graph import Graph
import tensorflow as tf

def is_subgraph(graph, edges):
	return set(edges) <= set(graph.edge_dict.keys())

def is_tree(num_vertices, edges):
	if len(edges) != 2 * (num_vertices - 1) and len(edges) != num_vertices - 1:
		print("incorrect number of edges")
		return False
	matrix = np.zeros((num_vertices, num_vertices))
	for i in range(num_vertices):
		for j in range(num_vertices):
			if (i, j) in edges:
				matrix[i, j] = 1
				matrix[j, i] = 1

	g = Graph(matrix)

	if len(g.get_connected_vertices(0)) != num_vertices:
		print("nope, not connected")
		return False
	return True

def is_cycle(matrix):
	# matrix is V x V np array. Check that every vertex is represented exactly once. 
	v = matrix.shape[0]
	zero = np.ones(v) == np.sum(matrix, axis=0)
	one = np.ones(v) == np.sum(matrix, axis=1)
	return zero and one

def distance(p1, p2):
	# p1, p2 are 2D tuples. Calculate euclidean distance
	return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

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
	eu_coords = [[e[0] + max_coord / 2, e[1] + max_coord / 2] for e in eu_coords]
	eu_coords.sort(key=lambda x: distance((0, 0), x))

	edges = np.zeros((vertices, vertices))
	for i in range(vertices):
		for j in range(vertices):
			if i == j:
				continue
			edges[i, j] = distance(eu_coords[i], eu_coords[j])

	return edges


def polar_to_euclid(polar):
	r = polar[0]
	theta = math.radians(polar[1])

	return [r * math.cos(theta), r * math.sin(theta)]

def pc_graph(num_vertices, max_coord, relative):
	# planar_connected_graph
	# generate points in 2D space, in first quadrant max_coord x max_coord
	# get distances between points to generate 2D planar graph. 
	# returns: edges -> a symmetric matrix of edge lengths
	coords = []
	for i in range(num_vertices):
		coords.append(max_coord * np.random.rand(2))

	if relative:
		coords.sort(key=lambda x: distance((0,0), x))

	edges = np.zeros((num_vertices, num_vertices))
	for i in range(num_vertices):
		for j in range(num_vertices):
			if i == j:
				continue
			edges[i, j] = distance(coords[i], coords[j])

	return edges

def non_pc_graph(num_vertices, max_dist, relative):
	edges = np.zeros((num_vertices, num_vertices))
	for i in range(num_vertices):
		for j in range(i + 1, num_vertices):
			edge_val = max_dist * np.random.rand()
			edges[i, j] = edge_val
			edges[j, i] = edge_val
	return edges

def eu_shortest_cycle(graph_matrix, in_matrix=True, with_transpose=True):
	# takes a edge matrix as given by planar_connected_graph
	# checks (n-1)! different paths by setting vertex n to be the last vertex seen before the first one
	# i.e. n = 5 -> permute(1-4), 5 -> 32415 

	# algorithm credits: http://gregorulm.com/finding-an-eulerian-path/

	graph = Graph(graph_matrix)
	mst = graph.get_mst()

	def freqencies():
		my_list = [x for (x, y) in mst]
		result = [0 for i in range(max(my_list) + 1)]
		for i in my_list:
			result[i] += 1
		return result
		 
	def find_node(tour):
		for i in tour:
			if freq[i] != 0:
				return i
		return -1
	 
	def helper(tour, next):
		find_path(tour, next)
		u = find_node(tour)
		while sum(freq) != 0:     
			sub = find_path([], u)
			tour = tour[:tour.index(u)] + sub + tour[tour.index(u) + 1:]  
			u = find_node(tour)
		return tour
				  
	def find_path(tour, next):
		for (x, y) in mst:
			if x == next:
				current = mst.pop(mst.index((x,y)))
				mst.pop(mst.index((current[1], current[0])))
				tour.append(current[0])
				freq[current[0]] -= 1
				freq[current[1]] -= 1
				return find_path(tour, current[1])
		tour.append(next)
		return tour             
			  
	freq = freqencies()  
	cycle = helper([], mst[0][0])
	if in_matrix:
		return cycle_matrix(cycle, with_transpose)
	return cycle

def ex_shortest_cycle(graph_matrix, in_matrix=True, with_transpose=True):
	num_vertices = graph_matrix.shape[0]
	perm_list = list(itertools.permutations(np.arange(num_vertices - 1).tolist()))

	cycle_lengths = []
	for i in range(len(perm_list)):
		verts = list(perm_list[i])
		verts.append(num_vertices - 1)
		cycle_lengths.append(cycle_cost(graph_matrix, verts)) 

	ind = cycle_lengths.index(min(cycle_lengths))
	cycle = list(perm_list[ind]) + [num_vertices - 1]
	if in_matrix:
		return cycle_matrix(cycle, with_transpose)
	return cycle

def cycle_cost(graph_matrix, cycle):
	num_vertices = graph_matrix.shape[0]
	cycle.append(cycle[0])
	total = 0
	for j in range(num_vertices):
		total += graph_matrix[cycle[j], cycle[j + 1]]
	return total 

def cycle_matrix(cycle, with_transpose=True):
	v = len(cycle)
	cycle.append(cycle[0])
	matrix = np.zeros((v, v))
	for i in range(v):
		matrix[cycle[i], cycle[i + 1]] = 1
	if with_transpose:
		matrix += matrix.T
	return matrix

