# Create latex equations
python3 -m readme2tex --output README.md README_RAW.md --nocdn

cp -r svgs/ docs/img

cp -r img/ docs/img

# Rebuild documentation
cd docs
cat ../README.md | sed 's/:\([a-z_]*\):/|:\1:|/g' > ./README_DOCS.md
sed -i "" 's/svgs\///g' README_DOCS.md
make clean && make html
cd ..
