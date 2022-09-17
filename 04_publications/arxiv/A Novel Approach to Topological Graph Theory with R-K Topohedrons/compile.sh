#!/bin/bash
BASENAME="demo"

echo "Making backup"
cp "${BASENAME}.tex" "backups/demo(date +%s).tex"

pdflatex "${BASENAME}.tex" -aux_directory=intfiles
biber "${BASENAME}"
pdflatex "${BASENAME}.tex" -aux_directory=intfiles
pdflatex "${BASENAME}.tex" -aux_directory=intfiles

echo "Making pdf copy"
cp ${BASENAME}.pdf "pdfs/${BASENAME}$(date +%s).pdf"
echo "Done. File is pdfs/${BASENAME}$(date +%s).pdf"


echo "Cleaning things up"
EXTARRAY=(aux bbl bcf blg log toc)
for ext in ${EXTARRAY[*]};
do
    if [ ! -f ${BASENAME}.${ext} ]; then
	continue
    fi
    echo "Moving ${BASENAME}.${ext}"
    mv ${BASENAME}.${ext} aux/
done;

if [ -f ${BASENAME}.run.xml ]; then
    mv ${BASENAME}.run.xml aux
fi

if [ -f texput.log ]; then
    mv texput.log aux
fi

echo "COMPLETE"
