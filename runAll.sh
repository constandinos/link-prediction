#!/bin/bash

#######################################################################################################################
# runAll.sh
# This is a bash script to run all experiments together.
#
# Execution commands:
# chmod +x runAll.sh
# ./runAll &
#
# Created by: Constandinos Demetriou, 2021
#######################################################################################################################

# declare an array with the name of datasets
declare -a StringArray=("hamsterster" "twitch" "github" "deezer" "facebook" "erdos")

# declare the number of CUPs for using
NUM_OF_CPU = 8
 
# iterate the datasets using for loop
for val in ${StringArray[@]}; do
	
	# make a directory for the results
	mkdir results	
	
	cd src/
	# process datasets
	python process_dataset.py $val".txt" > ../results/$val".out" 2> ../results/$val".err"
	# export features
	python featurization.py $val"_edges.csv" >> ../results/$val".out" 2>> ../results/$val".err"
	# classification using supervised learning
	python models.py $val"_edges_features.csv" $NUM_OF_CPU >> ../results/$val".out" 2>> ../results/$val".err"
	cd ..
	
	# rename directory with results
	mv "results" "results_"$val

done
