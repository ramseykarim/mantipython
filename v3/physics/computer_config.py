import os
# Variables specific to the current computer

# For INSTRUMENT

# Desktop
bandpass_directory = "/n/sgraraid/filaments/data/filterInfo_PlanckHerschel/"
p_RIMO = "/n/sgraraid/filaments/data/filterInfo_PlanckHerschel/HFI_RIMO_R3.00.fits"

if not os.path.isdir(bandpass_directory):
    # Laptop
    bandpass_directory = "/home/ramsey/Documents/Research/Filaments/Herschel_bands/"
    p_RIMO = "/home/ramsey/Documents/Research/Filaments/HFI_stuff/HFI_RIMO_R3.00.fits"
