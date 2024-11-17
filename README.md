# Quantized-Translation
Translation:  
ONNX -> Good performance | running under 300 MB physical RAM  
PTH -> Good performance | running under 1.5 GB physical RAM  
TF -> Difficult putting together

STT:  
Using Sherpa-Onnx for STT. Requires roughly 300-400 MB RAM  
Compiled using docker cross compile image:  
docker pull dockcross/linux-armv7  
docker run --rm dockcross/linux-armv7 > ./dockcross  
chmod +x ./dockcross  

TTS:  
Using Sherpa-Onnx for TTS. Requires roughly 300-400 MB RAM
