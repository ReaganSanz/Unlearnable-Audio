# Progress  
New min-min function implemented. Noise sample saved is returned from this function for comparison with clean sample (NOTE: See issue 2)

# Issues/Incomplete  
1) Still not entirely sure if the min-min function is correct
2) Method of saving noisy graph is not working. (CURRENTLY FIXING)
3) Need to test different values for epsilon, step_size, etc. Current tests have it creating noise, however it is still noticable.  
4) All code is contained in one file, terrible formatting  
5) Model is generated everytime code is run  


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
