# Progress  
It appears like the Clean and Noisy audio have imperceptible differences (Please see NEW requirement below)  

# Issues/Incomplete  
1) Need to be sure I am saving the noisy and clean audio correctly for comparison
2) Need to test other levels of epsilon and accuracy threshold
3) All code is contained in one file, terrible formatting
4) Model is generated everytime code is run



# Requirements
Also Requires:

1) *NEW* sample_clean/ and sample_noisy/ directories to hold audio examples
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
