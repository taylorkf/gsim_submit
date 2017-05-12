#!/bin/bash

USER=$1
REL_1=$2
REL_2=$3
REL_3=$4
REL_4=$5
REL_5=$6
RELATIONS=( $REL_1 $REL_2 $REL_3 $REL_4 $REL_5)
for i in 0 1 2 3 4
do
    echo "User" $USER "experiment" $i "relationship" ${RELATIONS[i]}
    ./run.sh "$USER" "$i" "${RELATIONS[i]}"
done

