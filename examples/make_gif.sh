# ffmpeg -i /Users/jerry/Dropbox/Projects/AssistRobotics/assistive-gym/images/210203/output_BedBathingJacoRobotPose-v11_00.png -r 10 -s 320x180 /Users/jerry/Dropbox/Projects/AssistRobotics/assistive-gym/images/210203/output_BedBathingJacoRobotPose-v11_00.gif


# ffmpeg -i /Users/jerry/Dropbox/Projects/AssistRobotics/assistive-gym/images/210203/output_BedBathingJacoHuman-v1_01.png -r 10 -s 320x180 /Users/jerry/Dropbox/Projects/AssistRobotics/assistive-gym/images/210203/output_BedBathingJacoHuman-v1_01.gif


#!/bin/bash
# src_dir="data/200413/interactive_proposal_divide_training_03_propose_04/visualize"
# target_dir="data/200413/interactive_proposal_divide_training_03_propose_04/gifs"

src_dir="/Users/jerry/Dropbox/Projects/AssistRobotics/assistive-gym/images/210107"
target_dir="/Users/jerry/Dropbox/Projects/AssistRobotics/assistive-gym/images/210107"


mkdir -p "$target_dir"
IFS=$(echo -en "\n\b")

for file in `find "$src_dir" -type f -name "*.png"`
do
    # src_file=`echo "${file/.mp4/.}"`
    tgt_file=`echo "${file/$src_dir/$target_dir}"`
    tgt_file=`echo "${tgt_file/.png/.gif}"`
    # echo "$tgt_file"
    # ffmpeg -i "$file" -r 10 "$tgt_file"
    ffmpeg -i "$file" -r 10 -s 320x180 "$tgt_file" -y
done


# ffmpeg -i "$filename".mp4 -r 10 "$filename".gif
