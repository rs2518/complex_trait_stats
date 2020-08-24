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

cd "$CWD"

echo "";
echo "-----------------------------------------------------------";
echo "";
echo "Done!";
echo "";
echo "-----------------------------------------------------------";
echo "";