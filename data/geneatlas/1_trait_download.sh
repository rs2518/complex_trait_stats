#!/usr/bin/bash

# 'wget' does not exist for macOS. Use 'curl -Os' (or create 'wget' alias)
shopt -s expand_aliases
source ~/.bash_profile

if [ $# -eq 0 ]; then
  echo ""
  echo "-----------------------------------------------------------";
  echo ""
  echo "No arguments provided. For using the script, please provide a list of the trait keys to download. For some trait keys, you may need to add the key between quotes. For example:";
  echo "";
  echo "   bash download.sh 50-0.0 '(46/47)-0.0'";
  echo "";
  echo "A summary list of trait keys can be obtained from: http://geneatlas.roslin.ed.ac.uk/traits-table/";
  echo "";
  echo "-----------------------------------------------------------";
  echo "";
  echo "";

  exit 1
fi

if [ $# -gt 10 ]; then
  echo ""
  echo "-----------------------------------------------------------";
  echo ""
  echo "To avoid overloading the server, please do not download more than 10 traits at once.";
  echo "";
  echo "-----------------------------------------------------------";
  echo "";
  echo "";

  exit 1
fi

echo "-----------------------------------------------------------";
echo "";
echo "Downloading traits for keys $@";
echo "";
for arg in "$@";
do
  echo "    $arg";
done
echo "";
echo "in the current folder from Gene ATLAS website...";
echo "";
echo "-----------------------------------------------------------";
echo "";
echo "";

nerrors=0;
for arg in "$@";
do
  trait="${arg//[\(\)\/]/_}";
  
  echo "Downloading $trait...";
  
  wget ftp://ftp.igmm.ed.ac.uk/pub/GeneATLAS/${trait}.v2.tar
  # tar xopf ${trait}.v2.tar
  if [ $? -ne 0 ];
    then nerrors=$((nerrors + 1));
    else echo "$trait downloaded successfully.";
  fi
  
done

echo "------------------";
echo "Download finished.";

if [ $nerrors -ne 0 ];
  then echo "WARNING: Errors happened while downloading $nerrors traits.";
fi
