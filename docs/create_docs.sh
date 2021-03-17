cat ../README.md | sed 's/:\([a-z_]*\):/|:\1:|/g' > ./README_DOCS.md
cp -r ../img/ ./img
cp -r ../svgs/ ./_images/
sed -i "" 's/svgs/\.\.\/\.\.\/\.\.\/svgs/g' README_DOCS.md
make clean && make html
