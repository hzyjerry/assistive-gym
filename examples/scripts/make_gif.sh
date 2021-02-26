

convert_gifs() {

    src_dir="/Users/jerry/Dropbox/Projects/AssistRobotics/assistive-gym/images/${folder}"

    IFS=$(echo -en "\n\b")

    #for image_folder in `find "$src_dir" -maxdepth 1 -type d`
    for image_folder in `ls "$src_dir"`
    do
        target_folder="$src_dir/$image_folder/gifs"
        mkdir -p $src_dir/$image_folder/gifs

        for file in `ls -f "$src_dir/$image_folder" | grep .png`
        do
            src_file=$src_dir/$image_folder/$file
            tgt_file=$src_dir/$image_folder/gifs/$file
            tgt_file=`echo "${tgt_file/.png/.gif}"`
            if [ ! -f "$tgt_file" ]; then
                ffmpeg -i "$src_file" -r 10 -s 320x180 "$tgt_file" -y
            fi
        done
    done

    # ffmpeg -i "$filename".mp4 -r 10 "$filename".gif

}




folder="$1" # i.e. 210217



convert_gifs
