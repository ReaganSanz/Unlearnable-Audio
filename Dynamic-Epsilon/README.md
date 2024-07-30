# Update:  
- Attempting to implement Epsilon Variables that vary depending on the volume of audio at different segments. 
(LAST UPDATED: 8:34 AM, 7/30/24)


# Progress  
1) Min-Min has been altered to apply different epsilon values to different segments within the samples


# Issues/Incomplete  
1) Need to look into using "energy" to determine epsilon value instead of using mean of the amplitude
2) Accuracy is still around ~25%. I want to reach 10%, while remaining inperceptable
3) Also look into: Masks, equations for finding optimal eps values, psychoacoustic features, etc


   
# Requirements (Unchanged from original Unlearnable Audio)
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
