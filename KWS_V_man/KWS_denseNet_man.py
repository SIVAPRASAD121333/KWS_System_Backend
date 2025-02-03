import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys, copy
import librosa
import soundfile
import code
import datetime
import numpy as np
import os
import math
from torch.autograd import Variable
from read_writehtk import htkread, writehtk
from scipy.signal import lfilter
import speechproc

SYSTEM_PATH='./KWS_V_man/Manipuri_Model/'

class Bottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)
        print('x',x.shape, 'out', out.shape)
        out = torch.cat((x, out), 1)

        return out

class BasicBlock(nn.Module):
    def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0):
        super(BasicBlock, self).__init__()

        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, growthRate, kernel_size=3,  padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)
       # print('x',x.shape, 'out', out.shape)
        out = torch.cat((x, out), 1)

        return out

class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,    bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet(nn.Module):
    def __init__(self, num_classes, depth=100, block=BasicBlock, dropRate=0, growthRate=12, compressionRate=2, in_channels=1):
        super(DenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3 if block == BasicBlock else (depth - 4) // 6

        self.growthRate = growthRate
        self.dropRate = dropRate

        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = growthRate * 2
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=3, padding=1,  bias=False)

        self.dense1 = self._make_denseblock(block, n)
        self.trans1 = self._make_transition(compressionRate)
        self.dense2 = self._make_denseblock(block, n)
        self.trans2 = self._make_transition(compressionRate)
        self.dense3 = self._make_denseblock(block, n)

        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(8136, num_classes)


        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks):
        layers = []
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate))

            self.inplanes += self.growthRate

        return nn.Sequential(*layers)



    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes

        return Transition(inplanes, outplanes)


    def forward(self, x):
        x = self.conv1(x) # x - >>32, 24, 40, 101
        x = self.trans1(self.dense1(x)) #[32, 48, 16, 16] 
        x = self.trans2(self.dense2(x)) #[32, 60, 8, 8]
        x = self.dense3(x)              #[32, 132, 8, 8]
        x = self.bn(x)                  #[32, 132, 8, 8]
        x = self.relu(x)                #[32, 132, 8, 8]
        x = self.avgpool(x)             #[32, 132, 1, 1]
        x = x.view(x.size(0), -1)       #[32, 132]
    
        x = self.fc(x)                  #[32, 35]
        return x

device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

torch.manual_seed(42)

model = DenseNet(num_classes=21)
model.to(device)
print(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

n = count_parameters(model)
print("Number of parameters: %s" % n)

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

def predict(outputs):
    probabilities = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    return predictions  

def melspectrogram(xdata, samplerate, n_fft, hop_length, win_length, n_mels):
     #S = librosa.feature.melspectrogram(y=xdata, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, win_length=win_length)
     #S=librosa.power_to_db(S)  #(40, 101) - (melfilter, frame)
     S = librosa.feature.mfcc(y=xdata, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, win_length=win_length, n_mfcc=n_mels, dct_type=2)
     return  torch.FloatTensor(S)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10
criterion = nn.CrossEntropyLoss()
 
def main(fn):
   PATH=SYSTEM_PATH+'Manipuri_c21_model_epoch_50.model'
   model.load_state_dict(torch.load(PATH, map_location="cpu"))

   model.eval()

   keyword = np.genfromtxt(SYSTEM_PATH+'Manipuri_keyWordList',dtype='str')

   n_fft, n_mels = 1024, 40
   xdata, samplerate = soundfile.read(fn)
   print(samplerate)
   xdata = np.squeeze(np.asanyarray(xdata))
   xdata = xdata/np.amax(np.abs(xdata))  # normalize the signal

   #rVAD fast
   winlen, ovrlen, pre_coef, nfilter, nftt = 0.025, 0.01, 0.97, 20, 512
   ftThres = 0.5
   opts=1
   vadThres=0.4

   ft, flen, fsh10, nfr10 =speechproc.sflux(xdata, samplerate, winlen, ovrlen, nftt)
   # --spectral flatness --
   pv01=np.zeros(nfr10)
   pv01[np.less_equal(ft, ftThres)]=1
   pitch=copy.deepcopy(ft)

   pvblk=speechproc.pitchblockdetect(pv01, pitch, nfr10, opts)
   # --filtering--
   ENERGYFLOOR = np.exp(-50)
   b=np.array([0.9770,   -0.9770])
   a=np.array([1.0000,   -0.9540])
   fdata=lfilter(b, a, xdata, axis=0)
    
   #--pass 1--
   noise_samp, noise_seg, n_noise_samp=speechproc.snre_highenergy(fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk)

   #sets noisy segments to zero
   for j in range(n_noise_samp):
      fdata[range(int(noise_samp[j,0]),  int(noise_samp[j,1]) +1)] = 0

   vad_seg=speechproc.snre_vad(fdata,  nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, vadThres)   

   td=(vad_seg*fsh10)/samplerate
   voice =[]
   flag = 0
   for ix in np.arange(vad_seg.shape[0]):
      st=int(np.floor(td[ix,0]*samplerate))
      et=int(np.floor(td[ix,1]*samplerate))
      if flag==0:
         voice = xdata[st:et]
      else:
          voice = np.vstack((voice, xdata[st:et]))
      flag = flag + 0    

   hop_length = int(samplerate/100) # 10ms frame shift
   win_length = int(np.floor(samplerate*30/1000)) #% 30ms frame length
   xdata = voice
   #make 1-sec duration
   if len(xdata) < samplerate:
            zero_padding = np.zeros(samplerate - xdata.shape[0])
            xdata = copy.deepcopy(np.hstack((xdata, zero_padding)))
   else:
            xdata = copy.deepcopy(xdata[:samplerate])
   X = melspectrogram(xdata, samplerate, n_fft, hop_length, win_length, n_mels)
   input_ids = X.to(device)
   outputs = model(input_ids.unsqueeze(0).unsqueeze(0).permute(0,1,3,2)) ##torch.Size([1, 1, 98, 40])
   outputs = torch.softmax(outputs, dim=1)
   predictions  = predict(outputs)
   print('Identified Keyword:', keyword[predictions.cpu().detach().numpy()[0]]  )
   
   outputs = outputs.cpu().detach().numpy()
   #print('Details scores')
   #print('.................................')
   #for i in np.arange(0, outputs.shape[1]):
   #    print('%s: %.2f' %(keyword[i], outputs[0][i]))

   return (keyword[predictions.cpu().detach().numpy()[0]].split(',')[0],str(outputs[0][predictions.cpu().detach().numpy()[0]]))

#if __name__ == "__main__":
#   main()

