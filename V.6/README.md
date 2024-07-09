# Update:  
-Sampling Rate Variable Fixed  
(LAST UPDATED: 10:12 AM, 7/9/24)


# Progress  
1) Shuffling Problem Fixed: The train_loader was being randomly shuffled- leading to mismatched noise + samples in trainPerturb.py (perturbation.pt was holding different orders each time)
2) Indentation Problem Fixed: One of the for loops was supposed to be nested, it was accidentally not indented correctly
3) On a somewhat noisy test, it had around 20% accuracy- indicacting some level of unlearning due to noise generation.
4) NOTE: I am currently using loss instead of accuracy for threshold. And MAKE SURE seed value is the same in both programs  



# Issues/Incomplete  
1) Need to test on lower levels of noise (it is perceptable right now) to see if its still unlearnable


   
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
