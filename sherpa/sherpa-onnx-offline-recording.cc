#include <stdio.h>
#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <rtaudio/RtAudio.h>
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/wave-reader.h"

constexpr unsigned int kSampleRate = 16000;
constexpr unsigned int kBufferFrames = 256;

std::vector<float> audioBuffer;

int audioCallback(void *outputBuffer, void *inputBuffer, unsigned int nFrames, double streamTime, RtAudioStreamStatus status, void *userData) {
  if(status){
    std::cerr << "Stream overflow" << std::endl;
  }

  if(inputBuffer){
    float *input = static_cast<float *>(inputBuffer);
    audioBuffer.insert(audioBuffer.end(), input, input + nFrames);
  }

  return 0;
}

int recordAudio(){
  bool isRecording = false;
  while (true) {
    std::string command;
    std::cout << "Enter start to recognize audio: ";
    std::cin >> command;
    RtAudio audio;
    if(audio.getDeviceCount() < 1){
      std::cerr << "No audio devices" << std::endl;
      return -1;
    }
    RtAudio::StreamParameters inputParams;
    inputParams.deviceId = audio.getDefaultInputDevice();
    inputParams.nChannels = 1;
    inputParams.firstChannel = 0;

    if (command == "start" && !isRecording) {
      std::cout << "located start command" << std::endl;
      try {
        try{
          audio.openStream(nullptr, &inputParams, RTAUDIO_FLOAT32, kSampleRate, const_cast<unsigned int*>(&kBufferFrames), &audioCallback);
          std::cout << "created stream" << std::endl;
          audio.startStream();
          std::cout << "started stream" << std::endl;
        } catch(std::exception &e) {
          std::cerr << "RtAudio error: " << std::endl;
          return -1;
        }
        
        isRecording = true;
        std::cout << "Recording started..." << std::endl;

        if (audio.isStreamRunning()) audio.stopStream();
          audio.closeStream();
      } catch (std::exception &e) {
        std::cerr << "RtAudio error during recording: " << std::endl;
        return -1;
      }
    }
    else if (command == "stop" &&isRecording) {
      try{
        if(audio.isStreamRunning()) {
          audio.stopStream();
          audio.closeStream();
        }

        isRecording = false;
        std::cout << "Recording stopped." << std::endl;
        return 1;
      }
      catch (std::exception &e){
        std::cerr << "RtAudio error during shutdown: " << std::endl;
        return -1;
      }
    }
    else if(command == "stop" && !isRecording) {
      std::cout << "No recording active" << std::endl;
    }
    else{
      std::cout << "Unknown command" << std::endl;
    }
  }        
}

int main(int32_t argc, char *argv[]) {
    const char *kUsageMessage = R"usage(
Persistent Speech Recognition with sherpa-onnx.

Usage:
./bin/sherpa-onnx-offline --whisper-encoder=/path/to/encoder.onnx \
--whisper-decoder=/path/to/decoder.onnx \
--tokens=/path/to/tokens.txt \
--num-threads=1
)usage";

    sherpa_onnx::ParseOptions po(kUsageMessage);
    sherpa_onnx::OfflineRecognizerConfig config;
    config.Register(&po);

    po.Read(argc, argv);
    fprintf(stderr, "%s\n", config.ToString().c_str());

    if (!config.Validate()) {
        fprintf(stderr, "Errors in config!\n");
        return -1;
    }

    fprintf(stderr, "Creating recognizer ...\n");
    sherpa_onnx::OfflineRecognizer recognizer(config);
    fprintf(stderr, "Recognizer created. Setting up RtAudio...\n");

    int recordingStatus = recordAudio();
    if(recordingStatus == -1){
      std::cerr << "Error with recording" << std::endl;
      return(-1);
    }

    while (true) {
        std::string command;
        std::cout << "Enter process to recognize audio or exit to quit";
        std::cin >> command;

        if (command == "exit") {
            break; // Exit the loop and end the program
        }
        if(audioBuffer.empty()){
          std::cout << "No audio captured.";
        }

        const auto begin = std::chrono::steady_clock::now();

        auto s = recognizer.CreateStream();
        s->AcceptWaveform(kSampleRate, audioBuffer.data(), audioBuffer.size());

        sherpa_onnx::OfflineStream *ss[] = {s.get()};
        recognizer.DecodeStreams(ss, 1);

        std::cout << s->GetResult().AsJsonString() << "\n----\n";
        const auto end = std::chrono::steady_clock::now();
        float elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() / 1000.0;

        fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds);
        audioBuffer.clear();
    }

    fprintf(stderr, "Exiting program.\n");
    return 0;
}

