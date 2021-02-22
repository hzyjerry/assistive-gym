download_data() {
    gsutil -o 'GSUtil:parallel_process_count=1' -o 'GSUtil:parallel_thread_count=16' -m cp -r "gs://assistive-gym-experiments/rss-logs/logs/output/$folder" "new_models"
    echo $folder
}




folder="$1" # i.e. 210217



download_data
