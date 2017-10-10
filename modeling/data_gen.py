import numpy as np 
import math
import heapq
import itertools
from pprint import pprint
from graph_utils import *
from graph import Graph

def gen_and_save_data(num_vertices, max_coord, metric, symmetric, num_shards, shard_size, approx, prefix):
	# generate num_shards files of graphs, each having shard_size graphs. 
	# graphs will be uniform in characteristics of num_vertices, max_coord, metric
	label_func = eu_shortest_cycle if approx else ex_shortest_cycle
	gen_func = pc_graph if metric else non_pc_graph

	for i in range(1, num_shards + 1):
		data_array = np.zeros((shard_size, num_vertices ** 2))
		label_array = np.zeros((shard_size, num_vertices ** 2))
		for j in range(shard_size):
			g = gen_func(num_vertices, max_coord)
			data_array[j] = g.reshape(1, num_vertices ** 2)
			label_array[j] = label_func(g, True, symmetric).reshape(1, num_vertices ** 2)
		np.savetxt(prefix + "_data_%4d_%4d.csv" % (i, num_shards), data_array, delimiter=',')
		np.savetxt(prefix + "_labels_%4d_%4d.csv" % (i, num_shards), label_array, delimiter=',')


gen_and_save_data(5, 1, True, True, 1, 4096, False, "../graph_data/5v")