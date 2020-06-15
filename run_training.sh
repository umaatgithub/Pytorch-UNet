#!/bin/bash

INPUT='config/training_parameters.csv'
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read optimizer lr loss epochs batch cpname trainflag
do
	[ "$trainflag" -eq 0 ] && continue
	echo "Execute training with configuration parameters"
	echo "----------------------------------------------" 
	echo "Optimizer : $optimizer"
	echo "Learning rate : $lr"
	echo "Loss : $loss"
	echo "Epochs : $epochs"
	echo "Batch : $batch"
	echo "Checkpoint name : $cpname"
	
	command  python train.py -o $optimizer -lr $lr -l $loss -e $epochs -b $batch -cpn $cpname

done < $INPUT
IFS=$OLDIFS
