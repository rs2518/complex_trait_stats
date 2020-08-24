#!/usr/bin/bash

# 'wget' does not exist for macOS. Use 'curl -Os' (or create 'wget' alias)
shopt -s expand_aliases
source ~/.bash_profile

echo "";
echo "-----------------------------------------------------------";
echo "";
echo "Downloading GWAS 'variant info' files into current directory";
echo "";
echo "";
echo "The 'variant info' files can be downloaded manually from the download page of any trait. For example:";
echo "";
echo "The following link extracts the files from the 'J40-J47 Chronic lower respiratory diseases' trait";
echo "";
echo "http://geneatlas.roslin.ed.ac.uk/downloads/?traits=0";
echo "";
echo "-----------------------------------------------------------";
echo "";
echo "";


file=({1..22})
file+=("X")

prefix="http://static.geneatlas.roslin.ed.ac.uk/gwas/allWhites/snps/extended"
ext=".gz"
for i in ${file[@]};
do
  wget "$prefix/snps.imputed.chr$i.csv$ext"
done
wget "$prefix/hla/hla.freq$ext"

echo "";
echo "-----------------------------------------------------------";
echo "";
echo "Done!";
echo "";
echo "-----------------------------------------------------------";
echo "";

# prefix="http://statis.geneatlas.roslin.ed.ac.uk/gwas/allWhites/snps/extended"
# ext=".gz"
# for i in ${file[@]};
# do
#   printf '%s\n' "$prefix/snps.imputed.chr$i.csv$ext"
# done

# printf '%s\n' "$prefix/hla/hla.freq$ext"
# # wget "$prefix/hla/hla.freq$ext"

