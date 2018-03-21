# CGRU_layers

This repo contains files to train a neural gpu model to solve instances of the traveling salesman problem. 

* ```graph.py``` - Graph object and associated methods
* ```graph_utils.py``` - methods to find shortest cycles and also generate curriculum in the form of "cluster graphs"
* ```data_gen.py``` - makes graphs and stores them in folder ```graph_data``` (you'll need to make this empty directory beforehand)
* ```model_utils.py``` - methods relevant to the training of neural gpu
* ```test2.py``` - main training script. Loads curriculum made by ```data_gen.py``` and trains neural gpu model on it.  
