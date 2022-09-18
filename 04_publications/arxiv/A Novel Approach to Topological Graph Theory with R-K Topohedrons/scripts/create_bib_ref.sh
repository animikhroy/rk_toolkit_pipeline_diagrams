for f in `find bib -type f`; do
    echo \\addbibresource{${f}} 
    
done
