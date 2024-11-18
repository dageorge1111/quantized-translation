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

Notes:  
- Translation
  - Using [MarianMT Model](https://huggingface.co/docs/transformers/en/model_doc/marian) for translation
  - Best process is to convert from Pytorch to ONNX and quantize from there
  - Tensorflow lite has unneccessary difficulty transferring from ONNX
  - Static Quantization brings down performance severely
  - Dynamic Quantization to int8 is ideal for performance and efficiency
- STT
  - Modified [Sherpa-Onnx](https://k2-fsa.github.io/sherpa/onnx/index.html) for Speech to Text
  - Always on model to cut down loading time
- TTS
  - Modified [Sherpa-Onnx](https://k2-fsa.github.io/sherpa/onnx/index.html) for Text to Speech
