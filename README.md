# Analysis of Audio Signals Using Linear Predictive Coding
This study deals with audio signal feature extraction in order to be used for speaker authentication using a neuronal network.
Specifically, the effectiveness of linear predictive coding (LPC) coefficients is examined.
The goal of this study is to explain how LPC coefficients can be extracted and to evaluate whether they can be used to differentiate between multiple speakers.
Therefore, the developed audio preprocessing (noise and silence removal, framing and windowing) and LPC extraction method is applied to samples of 10 speakers from the [data set](https://www.kaggle.com/datasets/vjcalling/speaker-recognition-audio-dataset?resource=download).
A simple neural network is then trained and tested with the extracted features.

## Results
The evaluation of the data set using the neural network resulted in a prediction accuracy of **70.54 percent**, showing a loss of 5.47.
Thus the effectiveness of LPC for speaker authentication is proven.

## Subsequent Studies
### User Authentication Using Voice Recognition
[![](https://img.shields.io/badge/github-sa--hs--lb--jb-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/DHBW-FN-TIT20/sa-hs-lb-jb)</br>
The results of this study form the basis for the subsequent student research project.
Within the project, LPC is combined with other speaker related audio features like mel frequency cepstral coefficients to create a neuronal network structure that is capable of authenticating speakers.
The main goal of the student research project is to improve the systems accuracy by variating the calculated coefficients as well as the structure of the neural network.

## Code
The evaluation process is implemented in python.
The code is located within the [code](code/) directory.

### File structure
Location | Description
--- | ---
[main.py](code/main.py) and [main.ipynb](code/main.ipynb) | Starting point containing the function calls used for the complete evaluation process.
[DatasetHandler/DatasetHandler.py](code/DatasetHandler/DatasetHandler.py) | Class used for accessing the data set's files.
[AudioPreprocessor/AudioPreprocessor.py](code/AudioPreprocessor/AudioPreprocessor.py) | Class implementing preprocessors for noise and silence removal, framing and windowing.
[FeatureExtractor/ExtractorInterface.py](code/FeatureExtractor/ExtractorInterface.py) | Interface for implementing extraction classes.
[FeatureExtractor/LPCExtractor.py](code/FeatureExtractor/LPCExtractor.py) | Class based on ExtractorInterface implementing the LPC algorithm.
[FeatureExtractor/FeatureExtractor.py](code/FeatureExtractor/FeatureExtractor.py) | Class implementing general feature extraction using the ExtractorInterface implementations.
[FeatureEvaluator/FeatureEvaluator.py](code/FeatureEvaluator/FeatureEvaluator.py) | Class implementing methods for generating the data set coefficients as well as creating and evaluating the neuronal network model.

### How to use
1. Install python and pip
2. Install the pip libraries: `librosa`, `numpy`, `tensorflow` and `noisereduce`
3. Execute the [main.py](code/main.py) file or [main.ipynb](code/main.ipynb) jupyter notebook

## Author
### Henry Schuler
[![](https://img.shields.io/badge/github-schuler--henry-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/schuler-henry)
[![](https://img.shields.io/badge/E--Mail-contact@henryschuler.de-%23121011.svg?style=for-the-badge)](mailto:contact@henryschuler.de?subject=[GitHub]%20analysis-of-audio-signals-using-linear-predictive-coding)

## [LICENSE](LICENSE)
Copyright (c) 2022 Henry Schuler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
