cat ../README.md | sed 's/:\([a-z_]*\):/|:\1:|/g' > ./README_DOCS.md
make clean && make html