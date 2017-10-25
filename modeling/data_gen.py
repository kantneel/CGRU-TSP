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
		if j % 100 == 0:
			print(j)
		g = gen_func(num_vertices, max_coord, relative)
		zero_g = -1 * np.ones((border_vertices, border_vertices))
		zero_g[:num_vertices, :num_vertices] = g 
		data_array[j] = zero_g.reshape(1, border_vertices ** 2)

		l = label_func(g, True, symmetric)
		zero_l = (1 + (int(symmetric))) * np.eye(border_vertices)
		zero_l[:num_vertices, :num_vertices] = l
		label_array[j] = zero_l.reshape(1, border_vertices ** 2)
	print("HELLSDLFKJSDLFIJ")
	np.savetxt(prefix + "_data.csv", data_array, delimiter=',')
	np.savetxt(prefix + "_labels.csv", label_array, delimiter=',')

def gen_curriculum(num_vertices, border_vertices, max_coord, shard_size, approx, 
					n_clusters, max_pop, p_rad, p_arc, prefix):
	
	label_func = eu_shortest_cycle if approx else ex_shortest_cycle

	data_array = np.zeros((shard_size, border_vertices ** 2))
	label_array = np.zeros((shard_size, border_vertices ** 2))
	for j in range(shard_size):
		g = cluster_graph(num_vertices, max_coord, n_clusters, max_pop, p_rad, p_arc)
		zero_g = -max_coord / 15 * np.ones((border_vertices, border_vertices))
		zero_g[:num_vertices, :num_vertices] = g 
		data_array[j] = zero_g.reshape(1, border_vertices ** 2)

		l = label_func(g, True, True)
		zero_l = 2 * np.eye(border_vertices)
		zero_l[:num_vertices, :num_vertices] = l
		label_array[j] = zero_l.reshape(1, border_vertices ** 2)
	print("HELLSDLFKJSDLFIJ")
	np.savetxt(prefix + "_data.csv", data_array, delimiter=',')
	np.savetxt(prefix + "_labels.csv", label_array, delimiter=',')
	

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


#gen_and_save_data(5, 10, 100, 50008, True, True, True, False, "../graph_data/5v")
#gen_sharpening_data(10, 1, 40000, "../sharpening/10v")

gen_curriculum(8, 15, 100, 10000, True, 3, 3, 0.99, 0.4, "../graph_data/8_0")
gen_curriculum(8, 15, 100, 10000, True, 3, 3, 0.95, 0.5, "../graph_data/8_1")
gen_curriculum(8, 15, 100, 10000, True, 3, 3, 0.8, 0.2, "../graph_data/8_2")
gen_curriculum(8, 15, 100, 10000, True, 3, 3, 0.5, 0.2, "../graph_data/8_3")
gen_curriculum(8, 15, 100, 10000, True, 3, 3, 0.6, 0.4, "../graph_data/8_4")
gen_curriculum(8, 15, 100, 10000, True, 3, 3, 0.3, 0.6, "../graph_data/8_5")
#gen_and_save_data(8, 15, 100, 10000, True, True, True, False, "../graph_data/8_5")


