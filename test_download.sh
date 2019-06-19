#!/usr/bin/bash


# Preprocess Traits Table file in python
cd ~/MSc_project/Data/

path=$(pwd)
file='Traits\ Table\ \ GeneATLAS.csv'


# RUN PYTHON
python -c "import pandas as pd; df = pd.read_csv('$path$file',skiprows=1)" 

# or

#!/bin/python
import pandas as pd

table = pd.read_csv(path+'Traits Table  GeneATLAS.csv', skiprows = 1)





# Rename Traits Table csv file
mv Traits 

Traits\ Table\ \ GeneATLAS.csv




echo "Create array with traits to download"

echo "for loop to run through every 10 traits"

echo "bash download.sh <TRAIT ARRAY HERE!>"

 
