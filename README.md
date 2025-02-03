# Main Commands
cd KWS_System_Backend
pip install -r requirements.txt
python my_app.py

# Introduction

This is a keyword spotting (KWS) system for three different languages, i.e., Bengali, Manipuri, and Mizo. A GUI is also developed for this system. After the installation of all the requirements, user can simply run the server (by using the command given below), and then by using a specific [link](https://localhost:8000/) (mentioned below in the Readme file) in the web browser user can use this KWS system.


# Installation

First, follow the below step to download Anaconda from the below link using ```wget```

```
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
```

Use ```chmod``` to access the installed file

```
chmod 775 Anaconda3-2024.02-1-Linux-x86_64.sh
```

Install the Anaconda

```
./Anaconda3-2024.02-1-Linux-x86_64.sh
```

Use the follwoing steps to create a new conda environment with ```Python``` version ```3.9```.

```
cd /dataspace/diskspace/Anaconda
conda activate your_environment_name
conda create --name your_environment_name python=3.9
```

Activate the created conda environment

```
conda activate your_environment_name
```

Install ```Pytorch``` using the command below

```
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c nvidia
```

Now, use the below given commands to install a few more required packages.

```
pip install tensorboard_logger  
pip install --upgrade "protobuf<=3.20.1" 
pip install librosa     
conda install anaconda::django 
pip install django pydub
sudo apt-get install ffmpeg 
sudo apt-get install libav-tools
pip install django-sslserver
```
If all the above steps are successfully completed, then the system is ready the use our model.



# Steps to run the system

Go to the directory ```KWSSystem``` using ```cd```

``` 
cd KWSSystem
```

Use the below command for running the serever.

```
python manage.py runsslserver 0.0.0.0:8000 --cert your_certificate.crt --key your_private.key 
```

Now, in a web browser open the below link and allow unsecured access in the browser. 

```
https://localhost:8000/
```


# Reference
The Dense-net implementation  is considered as per this [resource](https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/densenet.py).





