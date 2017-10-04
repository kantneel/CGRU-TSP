# Making graphs and whatnot
import numpy as np 
import math
import heapq
from pprint import pprint

class Graph(object):

	def __init__(self, graph_matrix=None, vertices=10, edge_prob=0.5, edge_range=(1, 10), vert_rep_len=None):
		if graph_matrix is None:
			self.vertices = vertices
			self.graph_matrix = np.zeros((vertices, vertices))
			self.edge_prob = edge_prob
			self.edge_range = edge_range
			# edge_range is inclusive
		else:
			self.graph_matrix = graph_matrix
			self.vertices = graph_matrix.shape[0]
			self.edge_range = (np.min(graph_matrix), np.max(graph_matrix))
			self.adjacency_list = [[] for _ in range(self.vertices)]

		self.adjacency_list = [[] for _ in range(self.vertices)]
		self.edge_dict = {}
		# edge_dict -> {(u, v) : [weight, binary vector representation]}
		self.reverse_edge_dict = {}
		# reverse_edge_dict -> {(binary vector representation) : weight}
		self.edge_list = []
		# edge_list -> [u, v, weight]
		self.bin_vert_dict = {}
		# vert_dict -> {v : binary representations}
		self.vert_rep_len = int(np.log2(vertices)) + 1
		if vert_rep_len is not None:
			self.vert_rep_len = vert_rep_len
		
		for i in range(vertices):
			vert_bin = list(map(int, [c for c in str(bin(i))[2:]]))
			full_vert_bin = [0 for _ in range(self.vert_rep_len - len(vert_bin))] + vert_bin
			self.bin_vert_dict[i] = list(full_vert_bin)

		if graph_matrix is None:
			for i in range(vertices):
				for j in range(i + 1, vertices):
					if np.random.rand() < self.edge_prob:
						edge_value = int(np.random.randint(edge_range[0], high=edge_range[1] + 1))
						self.graph_matrix[i, j] = edge_value
						self.graph_matrix[j, i] = edge_value

						self.edge_dict[(i, j)] = [edge_value, self.bin_vert_dict[i] + self.bin_vert_dict[j]]
						self.edge_dict[(j, i)] = [edge_value, self.bin_vert_dict[j] + self.bin_vert_dict[i]]

						self.reverse_edge_dict[tuple(self.bin_vert_dict[i] + self.bin_vert_dict[j])] = edge_value
						self.reverse_edge_dict[tuple(self.bin_vert_dict[j] + self.bin_vert_dict[i])] = edge_value

						self.adjacency_list[i].append(j)
						self.adjacency_list[j].append(i)
		else:
			for i in range(vertices):
				for j in range(vertices):
					if self.graph_matrix[i,j] != 0:
						edge_value = self.graph_matrix[i,j]
						self.edge_dict[(i, j)] = [edge_value, self.bin_vert_dict[i] + self.bin_vert_dict[j]]
						self.reverse_edge_dict[tuple(self.bin_vert_dict[i] + self.bin_vert_dict[j])] = edge_value
						self.adjacency_list[i].append(j)

		for edge, weight in self.edge_dict.items():
			self.edge_list.append([edge[0], edge[1], weight[0]])

	def get_connected_vertices(self, start):
		visited_set = set([start])
		to_explore = []
		for vert in self.adjacency_list[start]:
			visited_set.add(vert)
			to_explore.append(vert)
		while(len(to_explore) > 0):
			for vert in self.adjacency_list[to_explore[0]]:
				if vert not in visited_set:
					visited_set.add(vert)
					to_explore.append(vert)
			del to_explore[0]

		return list(visited_set)

	def set_edge(self, start, target, val=None):
		if val == None:
			val = np.random.randint(self.edge_range[0], high=self.edge_range[1] + 1)

		self.graph_matrix[start, target] = val
		self.edge_dict[(start, target)] = [val, self.bin_vert_dict[start] + self.bin_vert_dict[target]]
		self.reverse_edge_dict[tuple(self.bin_vert_dict[start] + self.bin_vert_dict[target])] = val
		self.adjacency_list[start].append(target)

	def delete_edge(self, start, target):
		if self.graph_matrix[start, target] == 0:
			return None

		self.graph_matrix[start, target] = 0
		self.edge_dict.pop((start, target), None)
		self.reverse_edge_dict.pop(tuple(self.bin_vert_dict[start] + self.bin_vert_dict[target]), None)
		self.adjacency_list[start].remove(target)

	def connect_vertices(self):
		while True:
			vert_num = np.random.randint(self.vertices)
			connected = self.get_connected_vertices(vert_num)
			if len(connected) != self.vertices:
				pool = list(set(range(self.vertices)) - set(connected))
				to_connect = pool[np.random.randint(len(pool))]
				val = np.random.randint(self.edge_range[0], high=self.edge_range[1] + 1)
				self.set_edge(vert_num, to_connect, val)
				self.set_edge(to_connect, vert_num, val)
			else:
				break

	def get_mst(self):
		self.connect_vertices()
		start = np.random.randint(self.vertices)

		queue = []
		out_of_queue = []
		data = []
		pred = {}
		distances = {}
		for v in range(self.vertices):
			distances[v] = 9999999
			pred[v] = None
		distances[start] = 0
		for v in range(self.vertices):
			item = [distances[v], v]
			heapq.heappush(queue, item)
			data.append(item)
		while queue:
			dist, vert = heapq.heappop(queue)
			out_of_queue.append(vert)
			for next_vert in self.adjacency_list[vert]:
				new_cost = self.edge_dict[(vert, next_vert)][0]
				if next_vert not in out_of_queue and new_cost < distances[next_vert]:
					pred[next_vert] = vert
					distances[next_vert] = new_cost
					data[next_vert][0] = new_cost
					heapq._siftdown(queue, 0, queue.index(data[next_vert]))

		edge_list = []
		for key, val in pred.items():
			if key == None or val == None:
				continue
			edge_list.append((key, val))
			edge_list.append((val, key))
		return edge_list

	def get_shortest_path(self, start, end, binary=True):
		queue = []
		distances = {start : 0}
		data = {}
		pred = {}
		visited_set = set([start])

		for vert in self.adjacency_list[start]:
			distances[vert] = self.edge_dict[(start, vert)][0]
			item = [distances[vert], start, vert]
			heapq.heappush(queue, item)
			data[vert] = item

		while queue:
			cost, parent, vert = heapq.heappop(queue)
			if not vert in visited_set:
				pred[vert] = parent
				visited_set.add(vert)
				if vert == end:
					break
				for next_vert in self.adjacency_list[vert]:
					if next_vert in distances:
						if distances[next_vert] > self.edge_dict[(vert, next_vert)][0] + distances[vert]:
							distances[next_vert] = self.edge_dict[(vert, next_vert)][0] + distances[vert]
							data[next_vert][0] = distances[next_vert]
							data[next_vert][1] = vert
							heapq._siftdown(queue, 0, queue.index(data[next_vert]))
					else:
						distances[next_vert] = self.edge_dict[(vert, next_vert)][0] + distances[vert]
						item = [distances[next_vert], vert, next_vert]
						heapq.heappush(queue, item)
						data[next_vert] = item

		return_list = [end]
		current = end
		while True:
			return_list.append(pred[current])
			current = pred[current]
			if current == start:
				break
		return_list = return_list[::-1]
		if not binary:
			return return_list, distances[end]
		else:
			bin_list = []
			for i in range(len(return_list) - 1):
				info = self.edge_dict[(return_list[i], return_list[i + 1])]
				bin_list.append(info[1] + [info[0]])
			return np.array(bin_list)


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

	if matrix.T is matrix:
		print("you made something weird")
		return False
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

def planar_connected_graph(num_vertices, max_coord):
	# generate points in 2D space, in first quadrant max_coord x max_coord
	# get distances between points to generate 2D planar graph. 
	# returns: edges -> a symmetric matrix of edge lengths
	coords = []
	for i in range(num_vertices):
		coords.append(max_coord * np.random.rand(2))

	edges = np.zeros((num_vertices, num_vertices))
	for i in range(num_vertices):
		for j in range(num_vertices):
			if i == j:
				continue
			edges[i, j] = distance(coords[i], coords[j])

	return edges













