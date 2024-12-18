#include <iostream>
#include <thread>
#include <atomic>
#include "RtAudio.h"
#include <vector>
#include <stdio.h>
#include <chrono>  // NOLINT
#include <string>
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/wave-reader.h"

std::atomic<bool> isRecording(false);
std::vector<int16_t> recordedSamples;

void recordAudio(RtAudio &audio)
{
    if (audio.getDeviceCount() < 1)
    {
        std::cerr << "No audio devices found!" << std::endl;
        return;
    }

    RtAudio::StreamParameters parameters;
    parameters.deviceId = audio.getDefaultInputDevice();
    parameters.nChannels = 1;
    parameters.firstChannel = 0;
    unsigned int sampleRate = 16000;
    unsigned int bufferFrames = 512;

    auto audioCallback = [](void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
                            double streamTime, RtAudioStreamStatus status, void *userData) -> int
    {
        if (status)
            std::cerr << "Stream overflow detected!" << std::endl;
        if (isRecording)
        {
            int16_t *buffer = static_cast<int16_t *>(inputBuffer);
            recordedSamples.insert(recordedSamples.end(), buffer, buffer + nBufferFrames);
        }
        return 0;
    };

    try
    {
        audio.openStream(nullptr, &parameters, RTAUDIO_SINT16,
                         sampleRate, &bufferFrames, audioCallback, nullptr);
        audio.startStream();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return;
    }

    std::cout << "Recording... Type 'stop' to end recording." << std::endl;
    while (isRecording)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    try
    {
        audio.stopStream();
        if (audio.isStreamOpen())
            audio.closeStream();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

void performSpeechToText(int argc, char *argv[])
{
    sherpa_onnx::ParseOptions po("");
    sherpa_onnx::OfflineRecognizerConfig config;
    config.Register(&po);

    po.Read(argc, argv);
    if (!config.Validate())
    {
        std::cerr << "Errors in config!" << std::endl;
        return;
    }

    std::cerr << "Creating recognizer ..." << std::endl;
    sherpa_onnx::OfflineRecognizer recognizer(config);

    std::cerr << "Started Speech-to-Text Processing" << std::endl;

    int32_t sampling_rate = 16000;
    std::vector<float> floatSamples;
    floatSamples.reserve(recordedSamples.size());
    for (auto sample : recordedSamples) {
        floatSamples.push_back(static_cast<float>(sample) / 32768.0f);
    }
    auto s = recognizer.CreateStream();
    s->AcceptWaveform(sampling_rate, floatSamples.data(), floatSamples.size());

    std::vector<sherpa_onnx::OfflineStream *> streams = {s.get()};
    recognizer.DecodeStreams(streams.data(), streams.size());

    std::cerr << "Done!" << std::endl;
    std::cout << "Recognition Result: " << s->GetResult().AsJsonString() << std::endl;
}

int main(int argc, char *argv[])
{
    RtAudio audio;
    std::string userInput;

    while (true)
    {
        std::cout << "Type 'start' to begin recording or 'exit' to quit: ";
        std::cin >> userInput;

        if (userInput == "start")
        {
            isRecording = true;
            std::thread recordingThread(recordAudio, std::ref(audio));

            while (true)
            {
                std::cout << "Type 'stop' to end recording: ";
                std::cin >> userInput;
                if (userInput == "stop")
                {
                    isRecording = false;
                    recordingThread.join();
                    std::cout << "Recording stopped." << std::endl;
                    performSpeechToText(argc, argv);
                    break;
                }
            }
        }
        else if (userInput == "exit")
        {
            break;
        }
    }

    return 0;
}

