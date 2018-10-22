#!/bin/bash

# Shell script to extract specific files from nexrad level III data
# Author: Yuping Lu
# Date: 10/22/2018

#search_dir='/home/ylk/data/HAS011196865'
search_dir='/home/ylk/data/processed/demo'

N0R_dir='/home/ylk/data/processed/N0R'
N0X_dir='/home/ylk/data/processed/N0X'
N0C_dir='/home/ylk/data/processed/N0C'
N0K_dir='/home/ylk/data/processed/N0K'
N0H_dir='/home/ylk/data/processed/N0H'

target_dir='/home/ylk/data/processed/final'

# remove files in target dir and clear n0r.txt n0x.txt n0c.txt n0k.txt n0h.txt
`rm "${target_dir}"/*`
> n0r.txt
> n0x.txt
> n0c.txt
> n0k.txt
> n0h.txt

for entry in "$search_dir"/*
do
    # extract files from tar.gz files
    tar -xzf "$entry" -C N0R --wildcards '*N0R*'
    tar -xzf "$entry" -C N0X --wildcards '*N0X*'
    tar -xzf "$entry" -C N0C --wildcards '*N0C*'
    tar -xzf "$entry" -C N0K --wildcards '*N0K*'
    tar -xzf "$entry" -C N0H --wildcards '*N0H*'
    
    # match N0H with other files and move them to final
    entries=`ls $N0H_dir`
    counter=0
    for entry in $entries
    do
        if [ $(($counter%15)) -eq 0 ]; then
            pattern="*${entry: -12}"
            if [[ `find $N0R_dir -name $pattern` && `find $N0X_dir -name $pattern` && `find $N0C_dir -name $pattern` && `find $N0K_dir -name $pattern` ]]; then
                echo "$entry" >> n0h.txt
                cp "${N0H_dir}/${entry}" "$target_dir"
                
                f1=`find $N0R_dir -name $pattern`
                n0r=$(basename ${f1[0]})
                echo "$n0r" >> n0r.txt
                cp "${N0R_dir}/${n0r}" "$target_dir"
    
                f2=`find $N0X_dir -name $pattern`
                n0x=$(basename ${f2[0]})
                echo "$n0x" >> n0x.txt
                cp "${N0X_dir}/${n0x}" "$target_dir"
    
                f3=`find $N0C_dir -name $pattern`
                n0c=$(basename ${f3[0]})
                echo "$n0c" >> n0c.txt
                cp "${N0C_dir}/${n0c}" "$target_dir"
                
                f4=`find $N0K_dir -name $pattern`
                n0k=$(basename ${f4[0]})
                echo "$n0k" >> n0k.txt
                cp "${N0K_dir}/${n0k}" "$target_dir"
            fi
        fi
        counter=$((counter+1))
    done
    
    # remove files in N0R etc.
    `rm "${N0R_dir}"/*`
    `rm "${N0X_dir}"/*`
    `rm "${N0C_dir}"/*`
    `rm "${N0K_dir}"/*`
    `rm "${N0H_dir}"/*`
done
