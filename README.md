Also Requires:

.env/
enviornment/
SpeechCommands/ (with 32 classes)

[name].sh with the following lines:
#!/bin/bash
 
#SBATCH --job-name=speech_command
#SBATCH --mem=16G ## memory that you need for your code
#SBATCH --gres=gpu:1 ## change this according to your need. It can go up to 4 GPUs.
#SBATCH --output=out%j.txt
#SBATCH --error=err%j.txt
 
source .env/bin/activate ## creating a virtual environment is a must
python3 speechNoise.py
