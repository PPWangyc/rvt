# Recurrent Video Transformer(RVT)
## Environment
pytorch(1.12.1), sklearn, numpy, python 3.8+

## Training
1. Save the pre-trained model under the checkpoint folder, Download [here](https://drive.google.com/file/d/1_ikuRGrHCO_kF6V_6boQZ03ZrU17EvWV/view?usp=sharing)
2. Place the Dataset and adjust the script path variable.
3. Go to the scripts folder
```
source train_rvt.sh
```
The script will automatically implement the Leave One Out Cross-Validation(LOOCV).
The running log file `train.log` will be output in log folder.
The experiment results are output in `output_rvt.json` in output folder