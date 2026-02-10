
 - [What is EMG Gesture Recognition ?](#What-is-EMG-Gesture-Recognition-?)
 - [Why EMG Gesture Recognition Matters ?](#Why-EMG-Gesture-Recognition-Matters-?)
-  [Setup](#Setup)
   - [Windows](#Windows)
   -  [Linux](#Linux)
   -  [MacOS](#MacOS)
## What is EMG Gesture Recognition ? 
When the neuromuscular system transmits electrical signals to contract muscles for movement, the detection and analysis of these signals is called _Electromyography (EMG)_. **_EMG Gesture Recognition_** refers to the decoding of neuromuscular signals to classify distinct hand gestures.
## Why EMG Gesture Recognition Matters ? 
>  **Human-Computer Interaction (HCI)**: Replaces physical controllers with natural muscle input, enabling seamless and immersive interaction in Virtual and Augmented Reality (XR).
 
>  **Prosthetics Control**: Decodes biological signals to drive bionic limbs, restoring dexterity and allowing amputees to control prosthetics intuitively.

>  **Rehabilitation**: Gamifies physical therapy by visualizing muscle engagement, helping patients track their recovery progress with real-time biofeedback.

>  **Accessibility**: Empowers individuals with limited mobility or motor impairments to control digital devices using even faint muscle signals.

## Setup

### Windows
##### Clone the repository
```sh
git clone https://github.com/YKesX/emg-ninapro-gesture-recognition.git
cd emg-ninapro-gesture-recognition
```
##### Create virtual environment and activate
```sh
python -m venv venv
.\venv\Scripts\activate
```
##### Install dependencies
```sh
pip install -r requirements.txt
pip install jupyter
```
##### Run
```sh
jupyter notebook
```
### Linux
##### Clone the repository
```sh
git clone https://github.com/YKesX/emg-ninapro-gesture-recognition.git
cd emg-ninapro-gesture-recognition
```
##### Create virtual environment and activate 
```sh
python3 -m venv venv
source venv/bin/activate
```
##### Install dependencies 
```sh
pip install --upgrade pip
pip install -r requirements.txt
pip install jupyter
```
##### Run
```sh
jupyter notebook
``` 
### MacOS
##### Clone the repository
```sh
git clone https://github.com/YKesX/emg-ninapro-gesture-recognition.git
cd emg-ninapro-gesture-recognition
```
##### Create virtual environment and activate
```sh
python3 -m venv venv
source venv/bin/activate
```
##### Install dependencies
```sh
pip install -r requirements.txt
pip install jupyter
```
##### Run
```sh
jupyter notebook
```


### Dataset

We used the NinaPro DB1 dataset for EMG gesture recognition.
Link: https://www.nina-pro.org/databases/db1/
Citation: C. Atzori, A. Gijsberts, C. Castellini, B. Caputo, F. Hager, et al., "Electromyography data for non-invasive naturally-controlled robotic hand prostheses," Scientific Data, vol. 1, Article 140053, 2014, doi: 10.1038/sdata.2014.53.