#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iterator>
#include <sstream>

// Helper function to load tokenizer metadata (vocabulary)
std::vector<std::string> load_vocab(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    std::vector<std::string> vocab;
    std::string line;
    while (std::getline(file, line)) {
        vocab.push_back(line);
    }
    return vocab;
}

// Function to tokenize the input text
std::vector<int64_t> tokenize(const std::string& text, const std::vector<std::string>& vocab) {
    std::vector<int64_t> input_ids;
    std::istringstream stream(text);
    std::string word;
    while (stream >> word) {
        auto it = std::find(vocab.begin(), vocab.end(), word);
        if (it != vocab.end()) {
            input_ids.push_back(std::distance(vocab.begin(), it));
        } else {
            input_ids.push_back(3); // Unknown token ID
        }
    }
    return input_ids;
}

// Helper function to pad inputs
void pad_sequence(std::vector<int64_t>& input, size_t max_length) {
    while (input.size() < max_length) {
        input.push_back(0); // Padding token ID
    }
    input.resize(max_length);
}

// Convert logits to token IDs (argmax)
int64_t argmax(const std::vector<float>& logits) {
    return std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
}

int main() {
    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Load ONNX model
    const std::string model_path = "../models/marianmt_quantized_static_tokenize.onnx";
    Ort::Session session(env, model_path.c_str(), session_options);

    // Load tokenizer vocabulary
    const std::string vocab_file = "../models/tokenizer/vocab.txt";
    auto vocab = load_vocab(vocab_file);

    // Input text
    std::string input_text = "Where is the nearest hotel?";
    size_t max_length = 64;

    // Tokenize input
    auto input_ids = tokenize(input_text, vocab);
    pad_sequence(input_ids, max_length);

    // Prepare ONNX inputs
    std::vector<int64_t> attention_mask(max_length, 1);
    size_t batch_size = 1;

    // Define the shape of the input tensor
    int64_t batch_size_int = static_cast<int64_t>(batch_size);
    std::array<int64_t, 2> input_shape = {batch_size_int, static_cast<int64_t>(max_length)};

    // Declare memory info
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create input tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size());

    Ort::Value attention_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size());

    // Decoder input
    int64_t decoder_start_token_id = 0; // Replace with actual start token ID
    std::vector<int64_t> decoder_input_ids = {decoder_start_token_id};

    // Run inference
    const char* input_names[] = {"input_ids", "attention_mask", "decoder_input_ids"};
    const char* output_names[] = {"logits"};

    size_t max_output_length = 30;
    std::vector<int64_t> output_ids;

    for (size_t i = 0; i < max_output_length; ++i) {
        // Pad the decoder_input_ids to match the expected max_length
        std::vector<int64_t> padded_decoder_input_ids = decoder_input_ids;
        pad_sequence(padded_decoder_input_ids, max_length);

        // Define the updated shape for the decoder input tensor
        std::array<int64_t, 2> new_decoder_shape = {batch_size_int, static_cast<int64_t>(max_length)};

        // Create the updated decoder tensor
        Ort::Value decoder_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, padded_decoder_input_ids.data(), padded_decoder_input_ids.size(), new_decoder_shape.data(), new_decoder_shape.size());

        // Prepare ONNX inputs using move semantics
        std::array<Ort::Value, 3> ort_inputs = {std::move(input_tensor), std::move(attention_tensor), std::move(decoder_tensor)};
        
        // Run inference
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, ort_inputs.data(), ort_inputs.size(), output_names, 1);

        // Get logits and find the next token
        auto* logits = output_tensors[0].GetTensorMutableData<float>();
        size_t vocab_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[2];

        std::vector<float> logits_vector(logits, logits + vocab_size);
        int64_t next_token_id = argmax(logits_vector);
        output_ids.push_back(next_token_id);

        // Stop if EOS token is encountered
        if (next_token_id == 1) { // Replace 1 with EOS token ID
            break;
        }

        // Update decoder input for next iteration
        decoder_input_ids.push_back(next_token_id);
        
        // Re-create the input tensor and attention tensor for subsequent iterations
        // Since input_tensor and attention_tensor were moved, recreate them for the next loop iteration.
        input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size());
        attention_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size());
    }

    // Decode the output
    std::string translated_text;
    for (auto id : output_ids) {
        if (id < vocab.size()) {
            translated_text += vocab[id] + " ";
        }
    }

    std::cout << "Translated Text: " << translated_text << std::endl;

    return 0;
}

