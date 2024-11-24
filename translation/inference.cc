#include <iostream>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>
#include <algorithm>

// Function to normalize input text
std::string normalize_text(const std::string& text) {
    std::string normalized = text;
    // Convert to lowercase (optional, depends on the tokenizer)
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
    // Trim whitespace
    normalized.erase(0, normalized.find_first_not_of(" \t\n\r"));
    normalized.erase(normalized.find_last_not_of(" \t\n\r") + 1);
    return normalized;
}

int main() {
    // Paths to the tokenizer and ONNX model
    std::string tokenizer_model_path = "../models/tokenizer/source.spm";
    std::string quantized_model_path = "../models/marianmt_model_int8.onnx";

    // Load the SentencePiece tokenizer
    sentencepiece::SentencePieceProcessor tokenizer;
    if (!tokenizer.Load(tokenizer_model_path).ok()) {
        std::cerr << "Failed to load the tokenizer model." << std::endl;
        return -1;
    }

    // Retrieve special token IDs
    int bos_token_id = tokenizer.PieceToId("<s>");
    int eos_token_id = tokenizer.PieceToId("</s>");
    int pad_token_id = tokenizer.PieceToId("<pad>");

    if (bos_token_id == -1 || eos_token_id == -1 || pad_token_id == -1) {
        std::cerr << "Special token IDs not defined in the tokenizer." << std::endl;
        return -1;
    }

    // Debug: Print special token IDs
    std::cout << "BOS token ID: " << bos_token_id << std::endl;
    std::cout << "EOS token ID: " << eos_token_id << std::endl;
    std::cout << "PAD token ID: " << pad_token_id << std::endl;

    // Initialize ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MarianMT");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Load the ONNX model
    Ort::Session session(env, quantized_model_path.c_str(), session_options);

    // Create memory info
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Preprocess the input text
    std::string input_text = "Why do you like Germany so much?";
    input_text = normalize_text(input_text);

    // Add special tokens
    input_text = "<s> " + input_text + " </s>";

    // Tokenize the input text
    std::vector<int> input_ids;
    if (!tokenizer.Encode(input_text, &input_ids).ok()) {
        std::cerr << "Failed to encode the input text." << std::endl;
        return -1;
    }

    // Debug: Print input token IDs
    std::cout << "Input text: " << input_text << std::endl;
    std::cout << "Input token IDs:" << std::endl;
    for (auto id : input_ids) {
        std::cout << id << " ";
    }
    std::cout << std::endl;

    // Decode and print the input tokens
    std::string decoded_input;
    if (!tokenizer.Decode(input_ids, &decoded_input).ok()) {
        std::cerr << "Failed to decode input tokens." << std::endl;
        return -1;
    }
    std::cout << "Decoded input tokens: " << decoded_input << std::endl;

    // Pad/truncate input to max_length
    const int max_length = 64;
    if (input_ids.size() > max_length) {
        input_ids.resize(max_length);
    } else {
        input_ids.insert(input_ids.end(), max_length - input_ids.size(), pad_token_id);
    }

    // Create attention mask
    std::vector<int64_t> attention_mask(max_length, 0);
    for (size_t i = 0; i < input_ids.size(); ++i) {
        attention_mask[i] = (input_ids[i] != pad_token_id) ? 1 : 0;
    }

    // Convert input_ids to int64_t for ONNX Runtime
    std::vector<int64_t> input_ids_int64(input_ids.begin(), input_ids.end());

    // Prepare decoder input IDs with the start token
    int decoder_start_token_id = bos_token_id;  // Use BOS token as the start token

    const int batch_size = 1;
    const int max_output_length = 30;  // Define the maximum output sequence length

    // Initialize outputs with padding
    std::vector<std::vector<int64_t>> outputs(batch_size, std::vector<int64_t>(max_length, pad_token_id));
    outputs[0][0] = decoder_start_token_id;  // Initialize with the start token

    // Perform inference in a loop for auto-regressive generation
    for (int i = 1; i < max_output_length && i < max_length; ++i) {
        // Prepare input shapes
        std::vector<int64_t> input_shape = {batch_size, static_cast<int64_t>(input_ids_int64.size())};
        std::vector<int64_t> decoder_input_shape = {batch_size, static_cast<int64_t>(i)};

        // Create tensors
        std::vector<Ort::Value> input_tensors;
        input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, input_ids_int64.data(), input_ids_int64.size(), input_shape.data(), input_shape.size()));

        input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size()));

        std::vector<int64_t> decoder_input_ids(outputs[0].begin(), outputs[0].begin() + i);
        input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, decoder_input_ids.data(), decoder_input_ids.size(), decoder_input_shape.data(), decoder_input_shape.size()));

        // Run the ONNX model
        const char* input_names[] = {"input_ids", "attention_mask", "decoder_input_ids"};
        const char* output_names[] = {"logits"};
        std::vector<Ort::Value> output_tensors = session.Run(
            Ort::RunOptions{nullptr}, input_names, input_tensors.data(), input_tensors.size(), output_names, 1);

        // Get the logits
        Ort::Value& logits_tensor = output_tensors[0];
        float* logits_data = logits_tensor.GetTensorMutableData<float>();
        std::vector<int64_t> logits_shape = logits_tensor.GetTensorTypeAndShapeInfo().GetShape();

        int64_t vocab_size = logits_shape[2];

        // Get the logits for the last generated token
        float* last_token_logits = logits_data + (i - 1) * vocab_size;

        // Get the next token ID (greedy decoding)
        int64_t next_token_id = std::distance(
            last_token_logits, std::max_element(last_token_logits, last_token_logits + vocab_size));
        outputs[0][i] = next_token_id;

        // Stop if EOS token is generated
        if (next_token_id == eos_token_id) {
            std::cout << "EOS token generated at step " << i << std::endl;
            break;
        }
    }

    // Debug: Print final output tokens
    std::cout << "Final output token IDs:" << std::endl;
    for (auto id : outputs[0]) {
        std::cout << id << " ";
    }
    std::cout << std::endl;

    // Decode the output tokens
    std::vector<int> generated_ids(outputs[0].begin(), outputs[0].end());
    std::string translation;
    if (!tokenizer.Decode(generated_ids, &translation).ok()) {
        std::cerr << "Failed to decode output tokens." << std::endl;
        return -1;
    }

    // Display the translation result
    std::cout << "Translation: " << translation << std::endl;

    return 0;
}

