## Check logs
gsutil ls -lh "gs://assistive-gym-experiments/rss-logs/" | sort -k 3



gsutil -o 'GSUtil:parallel_process_count=1' -o 'GSUtil:parallel_thread_count=16' -m cp -r "gs://assistive-gym-experiments/rss-logs/logs/output/ppo/BedBathing*" "new_models/ppo/"

gsutil -o 'GSUtil:parallel_process_count=1' -o 'GSUtil:parallel_thread_count=16' -m cp -r gs://assistive-gym-experiments/rss-logs/logs/output/ppo/BedBathingJacoHuman-v1/checkpoint_402 new_models/ppo/BedBathingJacoHuman-v1/


## SSH
gcloud compute ssh hzyjerry@user-study-icra20-1


## Copy local to remote

expdir="iterative_divide_initial_1v1_icra_14"
# expdir="iterative_divide_initial_1v1_icra_10"
zone="us-central1-a"
# zone="us-west2-a"
remote='user-study-icra20-1'
# remote='user-study-icra20-4'
# remote='user-study-icra20-7'
gcloud compute scp --zone "$zone" --recurse "data/201012/$expdir" "hzyjerry@$remote:~/study/rdb/examples/notebook/"


screen

cd study/rdb/examples/notebook && source activate studyenv
xvfb-run -s "-screen 0 1400x900x24" jupyter-notebook --no-browser --port=5000 --NotebookApp.token=abcd


python -m assistive_gym.learn --render --env BedBathingJacoHuman-v1 --seed 1 --load-policy-path new_models/ --render-eps 5 --colab


python -m assistive_gym.learn --render --env BedBathingJacoHumanPose-v14 --seed 1 --load-policy-path new_models/ --render-eps 5 --colab


python -m assistive_gym.learn --render --env BedBathingJacoRobotPose-v12 --seed 1 --load-policy-path new_models/ --render-eps 5 --colab


python -m assistive_gym.learn --render --env BedBathingJacoRobotPose-v13 --seed 1 --load-policy-path new_models/ --render-eps 5 --colab