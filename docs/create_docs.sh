cat ../README.md | sed 's/:\([a-z_]*\):/|:\1:|/g' > ./README_DOCS.md
cp -r ../svgs/ ./svgs
cp -r ../img/ ./img
make clean && make html