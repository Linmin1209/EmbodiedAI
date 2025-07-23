#!/bin/bash

for i in $(seq 0 29)
do
    echo "Running test with ID: $i"
    python tests/test_scene/test_agent.py $i
done 