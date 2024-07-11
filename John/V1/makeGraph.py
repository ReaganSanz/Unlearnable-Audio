import matplotlib.pyplot as plt

### Data
epochs = list(range(1, 11))  # Epochs from 1 to 10
accuracy_clean = [62.119160460392685, 74.13676371022342, 78.50372376438727, 80.36560595802302, 85.13879485443466, 83.24306025727827, 86.39133378469872, 87.98239675016926, 88.21936357481381, 88.79485443466486]
accuracy_008_noise = [15.301286391333784, 19.05890318212593, 13.676371022342586, 14.353419092755585, 16.68923493568043, 22.951929587000677, 14.861205145565336, 17.941773865944484, 16.486120514556532, 14.861205145565336]
accuracy_01_noise = [14.285714285714286, 13.879485443466486, 13.947190250507786, 14.454976303317535, 14.285714285714286, 12.931618144888287, 14.658090724441434, 13.981042654028435, 14.184157075152337, 14.116452268111036]
accuracy_025_noise = [11.712931618144887, 14.590385917400136, 14.285714285714286, 15.572105619498984, 18.38185511171293, 13.371699390656737, 11.137440758293838, 18.82193635748138, 21.49627623561273, 19.26201760324983]

### Transformation data with Epsilon at 0.01 and Low-Pass Filter
accuracy_lowpass_clean = [65.84292484766418, 77.69126607989168, 62.999322951929585, 84.63100880162492, 83.14150304671632, 84.29248476641841, 81.68584969532837, 85.44346648612051, 86.12051455653351, 89.0995260663507]
accuracy_lowpass_noise = [9.24170616113744, 14.319566689234936, 17.467840216655382, 19.63439404197698, 18.652674339878132, 22.004062288422478, 20.44685172647258, 15.301286391333784, 16.113744075829384, 17.366283006093433]

### Transformation data with Epsilon at 0.01 and Quantization
accuracy_quant_clean = [71.32701421800948, 79.75626269465133, 84.49559918754231, 84.76641841570752, 86.59444820582262, 83.68314150304671, 86.59444820582262, 86.76371022342586, 87.10223425863236, 88.35477318889642]
accuracy_quant_noise = [18.38185511171293, 15.030467163168584, 15.098171970209885, 11.509817197020988, 18.41570751523358, 18.788083953960733, 19.22816519972918, 18.551117129316182, 17.670954637779282, 17.53554502369668]

### Transformation data with Epsilon at 0.01 and Re-Sampling
accuracy_resample_clean = [69.66824644549763, 81.11035883547731, 84.25863236289777, 85.17264725795532, 87.16993906567366, 88.08395396073121, 88.82870683818551, 87.84698713608667, 87.88083953960731, 88.05010155721057]
accuracy_resample_noise = [16.418415707515233, 14.928909952606634, 16.756939742721734, 18.17874069058903, 16.079891672308733, 16.147596479350035, 21.428571428571427, 18.889641164522683, 16.181448882870683, 16.147596479350035]

# Plot 1: Clean Data
plt.figure(figsize=(10, 6))  # Adjust figure size as needed

# Epsilon Variation
plt.plot(epochs, accuracy_clean, 'o-', color='red', label='Clean')

# Transformation with Low-Pass Filter
plt.plot(epochs, accuracy_lowpass_clean, 's--', color='blue', label='Low-Pass Clean')

# Transformation with Quantization
plt.plot(epochs, accuracy_quant_clean, 'x-.', color='green', label='Quantization Clean')

# Transformation with Re-Sampling
plt.plot(epochs, accuracy_resample_clean, 'D:', color='purple', label='Re-Sample Clean')

# Customize plot
plt.ylim(60, 95)  # Adjust y-axis limits as needed
plt.title('Clean Data Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend(loc='lower right')
plt.tight_layout()

# Save or show plot
plt.savefig('results/acc_plot_clean.png')
plt.show()

# Plot 2: Noisy Data
plt.figure(figsize=(10, 6))  # Adjust figure size as needed

# Epsilon Variation
plt.plot(epochs, accuracy_008_noise, 'o-', color='lightcoral', label='0.008 Eps')
plt.plot(epochs, accuracy_01_noise, 'o-', color='red', label='0.01 Eps')
plt.plot(epochs, accuracy_025_noise, 'o-', color='darkred', label='0.025 Eps')

# Transformation with Low-Pass Filter
plt.plot(epochs, accuracy_lowpass_noise, 's--', color='blue', label='Low-Pass Noise')

# Transformation with Quantization
plt.plot(epochs, accuracy_quant_noise, 'x-.', color='green', label='Quantization Noise')

# Transformation with Re-Sampling
plt.plot(epochs, accuracy_resample_noise, 'D:', color='purple', label='Re-Sample Noise')

# Customize plot
plt.ylim(0, 25)  # Adjust y-axis limits as needed
plt.title('Noisy Data Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend(loc='lower right')
plt.tight_layout()

# Save or show plot
plt.savefig('results/acc_plot_noisy.png')
plt.show()

