#!/bin/zsh
set -e
mkdir -p outputs
netconvert -c project_intersection.netccfg
echo "Rete creata: project_intersection.net.xml"
echo "Test senza GUI:"
echo "  sumo -c project_intersection.sumocfg"
echo "Test con GUI:"
echo "  sumo-gui -c project_intersection.sumocfg"
