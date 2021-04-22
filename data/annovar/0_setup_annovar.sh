#!/usr/bin/bash

# 'wget' does not exist for macOS. Use 'curl -Os' (or create 'wget' alias)
shopt -s expand_aliases
source ~/.bash_profile

CWD=$(pwd)
DIRECTORY=$(dirname $0)

echo "";
echo "-----------------------------------------------------------";
echo "";
echo "Untarring 'annovar.tar'";
echo "";
echo "Ensure that Perl is successfully installed and ANNOVAR is downloaded on your system.";
echo "";
echo "If ANNOVAR is not yet downloaded, visit the ANNOVAR website and fill in the registration form.";
echo "";
echo "";
echo "See the following link for detailed step-by-step guide to set up ANNOVAR";
echo "";
echo "https://github.com/WGLab/Workshop_Annotation";
echo "";
echo "-----------------------------------------------------------";
echo "";
echo "";

cd "$DIRECTORY"
tar xvf annovar.tar

# # NOTE: 'exercise1.zip' file appears to be corrupted. Skip download
# echo "";
# echo "-----------------------------------------------------------";
# echo "";
# echo "Downloading 'exercise1.zip' file for practice setup";
# echo "";
# echo "-----------------------------------------------------------";
# echo "";
# echo "";

# wget https://github.com/WGLab/Workshop_Annotation/releases/download/v1.0.0/exercise1.zip
# unzip -a exercise1.zip

echo "";
echo "-----------------------------------------------------------";
echo "";
echo "Downloading databases (hg38 - build 150)";
echo "";
echo "";
echo "-----------------------------------------------------------";
echo "";
echo "";

cd "$DIRECTORY/annovar"
perl annotate_variation.pl -buildver hg38 -downdb -webfrom annovar refGene humandb/
perl annotate_variation.pl -buildver hg38 -downdb -webfrom annovar exac03 humandb/
perl annotate_variation.pl -buildver hg38 -downdb -webfrom annovar avsnp150 humandb/
perl annotate_variation.pl -buildver hg38 -downdb -webfrom annovar dbnsfp41a humandb/
perl annotate_variation.pl -buildver hg38 -downdb -webfrom annovar intervar_20180118 humandb/
perl annotate_variation.pl -buildver hg38 -downdb -webfrom annovar gene4denovo201907 humandb/
perl annotate_variation.pl -buildver hg38 -downdb -webfrom annovar gnomad30_genome humandb/
perl annotate_variation.pl -buildver hg38 -downdb -webfrom annovar 1000g2015aug_all humandb/
perl annotate_variation.pl -buildver hg38 -downdb -webfrom annovar clinvar_20210123 humandb/
perl annotate_variation.pl -buildver hg38 -downdb -webfrom annovar regsnpintron humandb/

cd "$CWD"

echo "";
echo "-----------------------------------------------------------";
echo "";
echo "Done!";
echo "";
echo "-----------------------------------------------------------";
echo "";
