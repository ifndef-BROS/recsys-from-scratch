/**
 * @file    data_loader.cpp
 * @brief   Implementation of CSV parsing routines declared in data_loader.h
 *
 * All parsing is done with standard C++ streams and string splitting —
 * no external CSV library is used, consistent with the STL-only constraint
 * of this project.
 *
 * Performance note:
 *   Loading ~14K items × 384 dims from CSV takes ~2-3s. This is a one-time
 *   startup cost. If cold-start time becomes a concern, consider serialising
 *   the embedding matrix to a binary format (e.g. raw float32 dump) which
 *   can be mmap'd directly into the embeddings vector.
 */

#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

using std::vector,
    std::string,
    std::stringstream,
    std::getline,
    std::unordered_map,
    std::ifstream,
    std::stoi,
    std::runtime_error,
    std::cout,
    std::stof;


/**
 * @brief Splits a CSV line on commas into a vector of string fields.
 *
 * Does not handle quoted fields or escaped commas. Safe for our generated
 * CSVs where ASINs and dimension values never contain commas.
 *
 * Declared static — internal linkage only, not part of the public API.
 *
 * @param line  A single raw line from a CSV file.
 * @return      Ordered vector of field strings.
 */
static vector<string> split_csv(const string &line) { // Why static?
    vector<string> fields;
    stringstream ss(line);
    string field;
    while (getline(ss, field, ','))
        fields.push_back(field);
    return fields;
}

void load_embeddings(
    const string &emb_path,
    const string &idx_path,
    embedding_t &embeddings,
    unordered_map<string, int> &asin_to_idx,
    unordered_map<int, string> &idx_to_asin
){
    {
        ifstream f(idx_path);
        if (!f.is_open())
            throw std::runtime_error("load_embeddings: cannot open index: " + idx_path);

        string line;
        getline(f, line); // to discard the field names
        while (getline(f, line)) {
            auto fields = split_csv(line);

            // Guard against blank trailing lines at EOF
            if (fields.size() < 2)
                continue;

            int idx = stoi(fields[1]);
            asin_to_idx[fields[0]] = idx;
            idx_to_asin[idx] = fields[0];
        }
    }

    embeddings.resize(asin_to_idx.size());
    {
        ifstream f(emb_path);
        if (!f.is_open())
            throw runtime_error("load_embeddings: cannot open embeddings: " + emb_path);

        string line;
        getline(f, line); // discard header: parent_asin,d0,d1,...,d383

        while (getline(f, line)) {
            auto fields = split_csv(line);

            // Each row must have: 1 asin + DIM float fields
            if (fields.size() < static_cast<size_t>(DIM + 1)) continue;
            int row = asin_to_idx.at(fields[0]);

            for (int d = 0; d < DIM; d++)
                embeddings[row][d] = std::stof(fields[d + 1]); // stof changes to float "100.8" -> 100.8
        }
    }

    cout << "[data_loader] embeddings : "
              << embeddings.size() << " items × " << DIM << " dims\n";
}

void load_train(
    const string& path,
    unordered_map<string, vector<Interaction>>& user_history
) {
    ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("load_train: cannot open: " + path);

    string line;
    getline(f, line); // discard header: user_id,parent_asin,rating,timestamp

    while (getline(f, line)) {
        auto fields = split_csv(line);
        if (fields.size() < 4) continue;

        // Timestamp (fields[3]) is intentionally ignored — chronological
        // order is already guaranteed by the Python sort in 03_filter_data.py.
        user_history[fields[0]].push_back({fields[1], std::stof(fields[2])});
    }
    std::cout << "[data_loader] train      : "
              << user_history.size() << " users\n";
}

void load_test(
    const std::string& path,
    std::unordered_map<std::string, std::string>& ground_truth
) {
    ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("load_test: cannot open: " + path);
    
    string line;
    getline(f, line); // discard header: user_id,parent_asin

    while (std::getline(f, line)) {
        auto fields = split_csv(line);
        if (fields.size() < 2) continue;
        ground_truth[fields[0]] = fields[1];
    }

    std::cout << "[data_loader] test       : "
              << ground_truth.size() << " users\n";
}