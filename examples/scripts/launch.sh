algo="ppo"


run_learn() {
    python assistive_gym/learn.py examples/params/${algo}_learn.yaml --train
}

launch_learn() {
    python examples/launch_training.py examples/params/${algo}_learn.yaml --train  --launch
}


run_eval() {
    python assistive_gym/learn.py examples/params/${algo}_learn.yaml --eval
}

# TODO: eval all envs in a directory
launch_eval() {
    python assistive_gym/learn.py examples/params/${algo}_learn.yaml --eval --launch
}


# TODO: render all envs in a directory
# TODO: bash cannot terminate
run_render() {
    echo "Rendering"
    python assistive_gym/learn.py examples/params/${algo}_learn.yaml --render
}


cmd="$1"

if [ -z cmd ]; then echo "cmd is unset" && exit 1 ; fi


if [[ $cmd == "learn" ]]
then
    run_learn;
elif [[ $cmd == "launch_learn" ]]
then
    launch_learn;
elif [[ $cmd == "eval" ]]
then
    run_eval;
elif [[ $cmd == "launch_eval" ]]
then
    launch_eval;
elif [[ $cmd == "render" ]]
then
    run_render;
fi
