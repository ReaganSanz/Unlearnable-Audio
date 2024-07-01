# Progress  
Ability to Train/Test the model on the unlearnable data is implemented (Updated last: 10:24 AM, 7/1/24)  
(NOTE: speechClass.py remains unchanged from V.4)

# Issues/Incomplete  
1) When running tests, the model IS LEARNING. With 4 epochs: 83% accuracy. Need to find out why. (is the issue trainPerturb.py or speechClass.py?)
2) Need to test different values for epsilon, step_size, etc. Is this why model is not unlearnable yet?
3) All code is contained in one file, terrible formatting

   

# Requirements
Also Requires:

1) sample_clean/ and sample_noisy/ directories to hold audio examples
2) .env/
3) enviornment/
4) SpeechCommands/ (I used 12 classes instead)
5) [name].sh with the following lines:

#!/bin/bash  
 
#SBATCH --job-name=speech_command  
#SBATCH --mem=16G ## memory that you need for your code  
#SBATCH --gres=gpu:1 ## change this according to your need. It can go up to 4 GPUs.  
#SBATCH --output=out%j.txt  
#SBATCH --error=err%j.txt  
 
source .env/bin/activate ## creating a virtual environment is a must  
python3 speechNoise.py
