#!/usr/bin/bash

# 'wget' does not exist for macOS. Use 'curl -Os' (or create 'wget' alias)
shopt -s expand_aliases
source ~/.bash_profile

CWD=$(pwd)
DIRECTORY=$(dirname $0)

# echo "";
# echo "-----------------------------------------------------------";
# echo "";
# echo "Downloading databases";
# echo "";
# echo "";
# echo "-----------------------------------------------------------";
# echo "";
# echo "";

cd "$DIRECTORY/annovar"
# annotate_variation.pl -buildver hg19 -downdb -webfrom annovar refGene humandb/
# annotate_variation.pl -buildver hg19 -downdb cytoBand humandb/
# annotate_variation.pl -buildver hg19 -downdb -webfrom annovar exac03 humandb/ 
# annotate_variation.pl -buildver hg19 -downdb -webfrom annovar avsnp147 humandb/ 
# annotate_variation.pl -buildver hg19 -downdb -webfrom annovar dbnsfp30a humandb/




# cd "$CWD"

# echo "";
# echo "-----------------------------------------------------------";
# echo "";
# echo "Done!";
# echo "";
# echo "-----------------------------------------------------------";
# echo "";