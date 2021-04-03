#!/bin/bash
 
# Declare an array of string with type
declare -a StringArray=("hamsterster" "github" "twitch" "deezer" "erdos")
 
# Iterate the string array using for loop
for val in ${StringArray[@]}; do
	
	mkdir results	
	cd src/
	
	python models.py $val"_edges_features.csv" 5 > ../results/$val".out" 2> ../results/$val".err"
	
	cd ..
	mv "results" "results_"$val
	
done
