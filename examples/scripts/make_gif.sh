

convert_gifs() {


    src_dir="/Users/jerry/Dropbox/Projects/AssistRobotics/assistive-gym/images/${folder}"
    target_dir="/Users/jerry/Dropbox/Projects/AssistRobotics/assistive-gym/images/${folder}/gifs"


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

}




folder="$1" # i.e. 210217



convert_gifs
