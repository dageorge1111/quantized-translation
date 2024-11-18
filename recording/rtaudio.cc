#include <iostream>
#include <string>
#include <rtaudio/RtAudio.h>
#include <cstdlib>

int record( void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames, double streamTime, RtAudioStreamStatus status, void *userData ){
  if (status) std::cout << "Stream overflow detected!" << std::endl;
  
  if(inputBuffer && wavFile.is_open()){
    
  }
}
