
run_learn() {
    algo="ppo"
    python assistive_gym/learn.py examples/params/${algo}_learn.yaml
}

launch_learn() {
    algo="ppo"
    python examples/launch_training.py examples/params/${algo}_learn.yaml  --launch
}


run_eval() {
    algo="ppo"
    out_file="$1"
    echo "Running eval to $out_file"
    python assistive_gym/eval.py examples/params/${algo}_eval.yaml | tee $out_file
}

# TODO: eval all envs in a directory
launch_eval() {
    algo="ppo"
    python assistive_gym/eval.py examples/params/${algo}_eval.yaml --launch
}


# TODO: render all envs in a directory
# TODO: bash cannot terminate
run_render() {
    echo "Rendering"
    algo="ppo"
    python assistive_gym/render.py examples/params/${algo}_render.yaml
}


cmd="$1"

if [ -z cmd ]; then echo "cmd is unset" && exit 1 ; fi


if [[ $cmd -eq "learn" ]]
then
    run_learn;
elif [[ $cmd -eq "launch_learn" ]]
then
    launch_learn;
elif [[ $cmd -eq "eval" ]]
then
    out_file="$2"
    run_eval $out_file;
elif [[ $cmd -eq "launch_eval" ]]
then
    launch_eval;
elif [[ $cmd -eq "render" ]]
then
    run_render;
fi