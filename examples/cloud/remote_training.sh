apt-get update --yes
apt-get upgrade --yes
apt-get install wget freeglut3-dev --yes
apt-get install git --yes
apt-get install zip unzip --yes

export PATH="$HOME/opt/git/bin:$PATH"

DISPLAY=':99.0'
Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &

apt-get install python3-pip --yes
pip install --upgrade pip
which git
which python3

## create env
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip3 install --upgrade pip
pip3 install -e ./assistive-gym

cd assistive-gym
wget https://www.dropbox.com/s/fw522y3zdi9kg9p/assets.zip
unzip assets.zip
mv assets assistive_gym/envs/assets
## Run experiment
cat examples/params/ppo_learn.yaml
python3 -m assistive_gym.learn --params_file examples/params/ppo_learn.yaml
