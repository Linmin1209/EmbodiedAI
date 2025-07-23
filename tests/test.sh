#!/bin/bash
cd /data1/linmin/EmbodiedAI
for i in $(seq 40 451)
do
    echo "Running test with ID: $i"
    python tests/test_scene/test_agent.py $i
done