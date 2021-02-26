

for folder in `ls new_models/210217/ppo`; do
    mkdir -p human_models/ppo/$folder
    cp -r new_models/210217/ppo/$folder/checkpoint_520 human_models/ppo/$folder/
    cp new_models/210217/ppo/$folder/params.yaml human_models/ppo/$folder/
done
#BedBathingJacoHuman-v0217_0-v1