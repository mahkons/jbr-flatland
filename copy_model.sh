#!/bin/bash

while true
do 
    echo "logpath:"
    read logpath
    if [ -d logdir/$logpath ]
    then
        break
    else
        echo "log doesn't exist"
    fi
done

cp logdir/"$logpath"/controller.torch generated/controller.torch
cp logdir/"$logpath"/experiment.torch generated/experiment.torch
cp logdir/"$logpath"/params.torch generated/params.torch
read v1 v2 v3 < tag.txt
v3=$((v3 + 1))
echo "$v1 $v2 $v3" > tag.txt


git add -f generated/controller.torch generated/experiment.torch generated/params.torch tag.txt
git commit -m "prepare submit"
git push

git tag "submission-v$v1.$v2.$v3"
git push origin "submission-v$v1.$v2.$v3"
