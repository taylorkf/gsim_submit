#!/bin/bash

uid=$1
eid=$2
rid=$3

declare -A spatial=(["0"]="up" ["1"]="down" ["2"]="left" ["3"]="right" ["4"]="on")

python ../src/sim.py ../worlds/user${uid}_${spatial[$rid]}.yaml ../data $uid $eid

