#!/bin/bash
 
# Declare an array of string with type
declare -a StringArray=("petster-hamster-household" "soc-hamsterster" "erdos" "musae" "facebook-wosn-links" )
 
# Iterate the string array using for loop
for val in ${StringArray[@]}; do
	python process_dataset.py $val".txt" > $val".out" 2> $val".err"
	python featurization.py $val"_edges.csv" >> $val".out" 2>> $val".err"
	python model.py $val"_edges_features.csv" >> $val".out" 2>> $val".err"
	
	mv "figures" "figures_"$val
	mv "results" "results_"$val
	mkdir figures
	mkdir results
done


