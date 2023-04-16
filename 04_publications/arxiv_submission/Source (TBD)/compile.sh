#!/bin/bash
BASENAME="novel_approach_to_topological_graph_theory_with_rk_diagrams_and_gravitational_wave_analysis"

echo "Making backup"

mkdir -p aux

biber "${BASENAME}"
pdflatex "${BASENAME}.tex" -aux_directory=intfiles -output-directory build

echo "Making pdf copy"

# echo "Cleaning things up"
# EXTARRAY=(aux bbl bcf blg log toc)
# for ext in ${EXTARRAY[*]};
# do
#     if [ ! -f ${BASENAME}.${ext} ]; then
# 	continue
#     fi
#     echo "Moving ${BASENAME}.${ext}"
#     mv ${BASENAME}.${ext} aux/
# done;

# if [ -f ${BASENAME}.run.xml ]; then
#     mv ${BASENAME}.run.xml aux/
# fi

# if [ -f texput.log ]; then
#     mv texput.log aux/
# fi

echo "COMPLETE"
