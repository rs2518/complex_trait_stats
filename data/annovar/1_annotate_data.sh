#!/usr/bin/bash

# 'wget' does not exist for macOS. Use 'curl -Os' (or create 'wget' alias)
shopt -s expand_aliases
source ~/.bash_profile

CWD=$(pwd)
DIRECTORY=$(dirname $0)


cd "$DIRECTORY/annovar"


# echo "";
# echo "-----------------------------------------------------------";
# echo "";
# echo "Downloading databases (hg19 - build 1)";
# echo "";
# echo "";
# echo "-----------------------------------------------------------";
# echo "";
# echo "";

perl table_annovar.pl ../ga_snps.txt humandb/ -buildver hg38 -out ga_anno -remove -protocol refGene,cytoBand,exac03,avsnp147,dbnsfp30a -operation gx,r,f,f,f -nastring - -csvout -polish -xref example/gene_fullxref.txt

# echo "";
# echo "-----------------------------------------------------------";
# echo "";
# echo "Annotating dataset";
# echo "";
# echo "";
# echo "-----------------------------------------------------------";
# echo "";
# echo "";

cd "$CWD"

echo "";
echo "-----------------------------------------------------------";
echo "";
echo "Done!";
echo "";
echo "-----------------------------------------------------------";
echo "";