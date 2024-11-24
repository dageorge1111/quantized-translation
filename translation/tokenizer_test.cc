#include <sentencepiece_processor.h>
#include <iostream>
#include <vector>
#include <string>

int main() {
    // Load the SentencePiece model
    sentencepiece::SentencePieceProcessor sp;
    if (!sp.Load("../models/tokenizer/source.spm").ok()) {
        std::cerr << "Failed to load SentencePiece model." << std::endl;
        return -1;
    }

    // Input text
    std::string input_text = "Why do you like Germany so much?";

    // Add special tokens
    int bos_token_id = sp.PieceToId("<s>");
    int eos_token_id = sp.PieceToId("</s>");
    int pad_token_id = sp.PieceToId("<pad>");

    std::vector<int> token_ids;
    token_ids.push_back(bos_token_id);
    if (!sp.Encode(input_text, &token_ids).ok()) {
        std::cerr << "Failed to tokenize input text." << std::endl;
        return -1;
    }
    token_ids.push_back(eos_token_id);

    // Apply truncation/padding
    const int max_length = 64;
    if (token_ids.size() > max_length) {
        token_ids.resize(max_length);
    } else {
        token_ids.insert(token_ids.end(), max_length - token_ids.size(), pad_token_id);
    }

    // Print token IDs
    std::cout << "Token IDs: ";
    for (int id : token_ids) {
        std::cout << id << " ";
    }
    std::cout << std::endl;

    // Decode token IDs back to text
    std::string decoded_text;
    std::vector<int> tokens_to_decode;
    for (int id : token_ids) {
        if (id != pad_token_id) { // Exclude padding tokens
            tokens_to_decode.push_back(id);
        }
    }
    if (!sp.Decode(tokens_to_decode, &decoded_text).ok()) {
        std::cerr << "Failed to decode token IDs." << std::endl;
        return -1;
    }

    std::cout << "Decoded text: " << decoded_text << std::endl;

    return 0;
}

