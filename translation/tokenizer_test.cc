#include <sentencepiece_processor.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // for transform

int main() {
    // Load the SentencePiece model
    sentencepiece::SentencePieceProcessor sp;
    if (!sp.Load("../models/tokenizer/source.spm").ok()) {
        std::cerr << "Failed to load SentencePiece model." << std::endl;
        return -1;
    }

    // Override token IDs to match Python behavior
    int pad_token_id = 58100; // Override manually
    int eos_token_id = 0;     // Consistent with Python
    int bos_token_id = -1;    // Not explicitly used in Python

    // Input text
    std::string input_text = "Why do you like Germany so much?";

    // Normalize text
    std::transform(input_text.begin(), input_text.end(), input_text.begin(), ::tolower); // Convert to lowercase
    std::cout << "Normalized Text: " << input_text << std::endl;

    // Add EOS token to the text (skip BOS token)
    std::string preprocessed_text = input_text + " </s>";
    std::cout << "Preprocessed Text: " << preprocessed_text << std::endl;

    // Tokenize the preprocessed text
    std::vector<int> token_ids;
    if (!sp.Encode(preprocessed_text, &token_ids).ok()) {
        std::cerr << "Failed to tokenize input text." << std::endl;
        return -1;
    }

    // Apply truncation/padding
    const int max_length = 64;
    if (token_ids.size() > max_length) {
        token_ids.resize(max_length);
    } else {
        token_ids.insert(token_ids.end(), max_length - token_ids.size(), pad_token_id);
    }

    // Print token IDs
    std::cout << "Token IDs: [";
    for (size_t i = 0; i < token_ids.size(); ++i) {
        std::cout << token_ids[i];
        if (i < token_ids.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Decode token IDs back to text
    std::string decoded_text;
    if (!sp.Decode(token_ids, &decoded_text).ok()) {
        std::cerr << "Failed to decode token IDs." << std::endl;
        return -1;
    }

    std::cout << "Decoded Text (including <pad>): " << decoded_text << std::endl;

    return 0;
}

