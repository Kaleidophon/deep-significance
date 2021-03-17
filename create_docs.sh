# Create latex equations
python3 -m readme2tex --output README.md README_RAW.md --nocdn

# Convert to .png and replace refs
cp -r svgs/ img/

for image_path in img/*.svg; do
  convert $image_path ${image_path/svg/png}  # This requires imagemagick
done

sed -i "" 's/svgs/img/g' README.md
sed -i "" 's/\.svg/\.png/g' README.md

# Rebuild documentation
cd docs
cat ../README.md | sed 's/:\([a-z_]*\):/|:\1:|/g' > ./README_DOCS.md
make clean && make html
cd ..
