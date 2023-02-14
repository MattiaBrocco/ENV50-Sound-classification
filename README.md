# Environmental-sounds-UNIPD-2022

*Dataset*: ESC50 (50 classes, 2000 examples).

*Preprocessing*: MFCCs, Chromagram, data augmentaion (7 times the initial sample size).

![eda_image](https://user-images.githubusercontent.com/61026948/218862548-b7420bb9-2630-4149-919b-91ac556eca5e.jpg)

*Evaluation metrics*: Accuracy, Estimated Memory Usage.
*Architectures*: CNN, RNN-SEQ2D, RNN431, RNN60-small, RNN60-LSTM, RNN60-GRU.

![RNN-1x60_8M_params](https://user-images.githubusercontent.com/61026948/218862612-18ab6ae9-c06a-40a6-aaa7-e1a8cd8a0a54.jpg)

*Best performing model*: in accuracy RNN60-LSTM (89.50% with 261.8 Mb), in accuracy with low memory usage RNN60-small (83.86% with 9.8 Mb).
