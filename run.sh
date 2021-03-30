#!/bin/bash
 
# Declare an array of string with type
declare -a StringArray=("hamsterster" "github" "deezer")
 
# Iterate the string array using for loop
for val in ${StringArray[@]}; do
	
	mkdir results	
	cd src/
	
	python process_dataset.py $val".txt" > ../results/$val".out" 2> ../results/$val".err"
	python featurization.py $val"_edges.csv" >> ../results/$val".out" 2>> ../results/$val".err"
	python models.py $val"_edges_features.csv" 7 >> ../results/$val".out" 2>> ../results/$val".err"
	
	cd ..
	mv "results" "results_"$val
	
done
