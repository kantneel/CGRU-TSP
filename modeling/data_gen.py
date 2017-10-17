import numpy as np 
import math
import heapq
import itertools
from pprint import pprint
from graph_utils import *
from graph import Graph

def gen_and_save_data(num_vertices, border_vertices, max_coord, shard_size, metric, symmetric, relative, approx, prefix):
	# generate num_shards files of graphs, each having shard_size graphs. 
	# graphs will be uniform in characteristics of num_vertices, max_coord, metric
	label_func = eu_shortest_cycle if approx else ex_shortest_cycle
	gen_func = pc_graph if metric else non_pc_graph

	data_array = np.zeros((shard_size, border_vertices ** 2))
	label_array = np.zeros((shard_size, border_vertices ** 2))
	for j in range(shard_size):
		g = gen_func(num_vertices, max_coord, relative)
		zero_g = -1 * np.ones((border_vertices, border_vertices))
		zero_g[:num_vertices, :num_vertices] = g 
		data_array[j] = zero_g.reshape(1, border_vertices ** 2)

		l = label_func(g, True, symmetric)
		zero_l = (1 + (int(symmetric))) * np.eye(border_vertices)
		zero_l[:num_vertices, :num_vertices] = l
		label_array[j] = zero_l.reshape(1, border_vertices ** 2)
	print("HELLSDLFKJSDLFIJ")
	np.savetxt(prefix + "_data_.csv", data_array, delimiter=',')
	np.savetxt(prefix + "_labels_.csv", label_array, delimiter=',')

def gen_sharpening_data(num_vertices, num_shards, shard_size, prefix):
	for i in range(1, num_shards + 1):
		data_array = np.zeros((shard_size, num_vertices ** 2))
		label_array = np.zeros((shard_size, num_vertices ** 2))
		for j in range(shard_size):
			num = np.random.randint(1, high=4)
			start = np.random.uniform(-num, num, (num_vertices, num_vertices))
			data_array[j] = start.reshape(1, num_vertices ** 2)

			softmaxed = np.exp(start) / np.sum(np.exp(start), axis=1)
			maxes = np.argmax(softmaxed, axis=1)
			new_array = np.zeros((num_vertices, num_vertices))
			for k in range(num_vertices):
				new_array[k, maxes[k]] = 1
			label_array[j] = new_array.reshape(1, num_vertices ** 2)
		np.savetxt(prefix + "_data_%s_%s.csv" % (i, num_shards), data_array, delimiter=',')
		np.savetxt(prefix + "_labels_%s_%s.csv" % (i, num_shards), label_array, delimiter=',')


gen_and_save_data(5, 10, 100, 50008, True, True, True, False, "../graph_data/5v")
#gen_sharpening_data(10, 1, 40000, "../sharpening/10v")