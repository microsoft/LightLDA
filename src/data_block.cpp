#include "data_block.h"
#include "document.h"
#include "common.h"

#include <multiverso/log.h>

#include <fstream>

#if defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#else 
#include <stdio.h>
#endif

namespace
{
    void AtomicMoveFileExA(std::string existing_file, std::string new_file)
    {
#if defined(_WIN32) || defined(_WIN64)
        MoveFileExA(existing_file.c_str(), new_file.c_str(),
            MOVEFILE_REPLACE_EXISTING);
#else 
        if (rename(existing_file.c_str(), new_file.c_str()) == -1)
        {
            multiverso::Log::Error("Failed to move tmp file to final location\n");
        }
#endif
    }
}

namespace multiverso { namespace lightlda
{
    DataBlock::DataBlock()
        : has_read_(false), num_document_(0), corpus_size_(0), vocab_(nullptr)
    {
        max_num_document_ = Config::max_num_document;
        memory_block_size_ = Config::data_capacity / sizeof(int32_t);

        documents_.resize(max_num_document_);
        
        try{
            offset_buffer_ = new int64_t[max_num_document_];
        }
        catch (std::bad_alloc& ba) {
            Log::Fatal("Bad Alloc caught: failed memory allocation for offset_buffer in DataBlock\n");
        }

        try{
            documents_buffer_ = new int32_t[memory_block_size_];
        }
        catch (std::bad_alloc& ba) {
            Log::Fatal("Bad Alloc caught: failed memory allocation for documents_buffer in DataBlock\n");
        }
    }

    DataBlock::~DataBlock()
    {
        delete[] offset_buffer_;
        delete[] documents_buffer_;
    }

    void DataBlock::Read(std::string file_name)
    {
        file_name_ = file_name;

        std::ifstream block_file(file_name_, std::ios::in | std::ios::binary);
        if (!block_file.good())
        {
            Log::Fatal("Failed to read data %s\n", file_name_.c_str());
        }
        block_file.read(reinterpret_cast<char*>(&num_document_), sizeof(DocNumber));

        if (num_document_ > max_num_document_)
        {
            Log::Fatal("Rank %d: Num of documents > max number of documents when reading file %s\n", 
                Multiverso::ProcessRank(), file_name_.c_str());
        }

        block_file.read(reinterpret_cast<char*>(offset_buffer_),
            sizeof(int64_t)* (num_document_ + 1));

        corpus_size_ = offset_buffer_[num_document_];

        if (corpus_size_ > memory_block_size_)
        {
            Log::Fatal("Rank %d: corpus_size_ > memory_block_size when reading file %s\n", 
                Multiverso::ProcessRank(), file_name_.c_str());
        }

        block_file.read(reinterpret_cast<char*>(documents_buffer_),
            sizeof(int32_t)* corpus_size_);
        block_file.close();

        GenerateDocuments();
        has_read_ = true;
    }

    void DataBlock::Write()
    {
        std::string temp_file = file_name_ + ".temp";

        std::ofstream block_file(temp_file, std::ios::out | std::ios::binary);

        if (!block_file.good())
        {
            Log::Fatal("Failed to open file %s\n", temp_file.c_str());
        }

        block_file.write(reinterpret_cast<char*>(&num_document_), 
            sizeof(DocNumber));
        block_file.write(reinterpret_cast<char*>(offset_buffer_),
            sizeof(int64_t)* (num_document_ + 1));
        block_file.write(reinterpret_cast<char*>(documents_buffer_),
            sizeof(int32_t)* corpus_size_);
        block_file.flush();
        block_file.close();

        AtomicMoveFileExA(temp_file, file_name_);
        has_read_ = false;
    }

    void DataBlock::GenerateDocuments()
    {
        for (int32_t index = 0; index < num_document_; ++index)
        {
            documents_[index].reset(new Document(
                documents_buffer_ + offset_buffer_[index],
                documents_buffer_ + offset_buffer_[index + 1]));
        }
    }
} // namespace lightlda
} // namespace multiverso
