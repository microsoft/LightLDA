#include "meta.h"
#include "common.h"

#include <fstream>
#include <multiverso/log.h>

namespace multiverso { namespace lightlda
{
    LocalVocab::LocalVocab() 
        : num_slices_(0), vocabs_(nullptr), size_(0), own_memory_(false)
    {}

    LocalVocab::~LocalVocab()
    {
        if (own_memory_)
        {
            delete[] vocabs_;
        }
    }

    AliasTableIndex::AliasTableIndex()
    {
        index_map_.resize(Config::num_vocabs, -1);
    }

    WordEntry& AliasTableIndex::word_entry(int32_t word)
    {
        if (index_map_[word] == -1)
        {
            Log::Fatal("Fatal in alias index: word %d not exist\n", word);
        }
        return index_[index_map_[word]];
    }

    void AliasTableIndex::PushWord(int32_t word,
        bool is_dense, int64_t begin_offset, int32_t capacity)
    {
        index_map_[word] = static_cast<int>(index_.size());
        index_.push_back({ is_dense, begin_offset, capacity });
    }

    Meta::Meta()
    {
    }

    Meta::~Meta()
    {
        for (int32_t i = 0; i < alias_index_.size(); ++i)
        {
            for (int32_t j = 0; j < alias_index_[i].size(); ++j)
            {
                delete alias_index_[i][j];
            }
        }
    }

    void Meta::Init()
    {
        tf_.resize(Config::num_vocabs, 0);
        local_tf_.resize(Config::num_vocabs, 0);
        int32_t* tf = new int32_t[Config::num_vocabs];
        int32_t* local_tf = new int32_t[Config::num_vocabs];
		local_vocabs_.resize(Config::num_blocks);
        for (int32_t i = 0; i < Config::num_blocks; ++i)
        {
            LocalVocab& local_vocab = local_vocabs_[i];

            std::string file_name = Config::input_dir 
                + "/vocab." + std::to_string(i);
            std::ifstream vocab_file(file_name, std::ios::in|std::ios::binary);

            if (!vocab_file.good())
            {
                Log::Fatal("Failed to open file : %s\n", file_name.c_str());
            }

            vocab_file.read(reinterpret_cast<char*>(&local_vocab.size_),
                sizeof(int));
            local_vocab.vocabs_ = new int[local_vocab.size_];
            local_vocab.own_memory_ = true;
            vocab_file.read(reinterpret_cast<char*>(local_vocab.vocabs_), 
                sizeof(int)*  local_vocab.size_);
            vocab_file.read(reinterpret_cast<char*>(tf), 
                sizeof(int)*  local_vocab.size_);
            vocab_file.read(reinterpret_cast<char*>(local_tf),
                sizeof(int)*  local_vocab.size_);

            vocab_file.close();

            for (int32_t i = 0; i < local_vocab.size_; ++i)
            {
                if (tf[i] > tf_[local_vocab.vocabs_[i]])
                {
                    tf_[local_vocab.vocabs_[i]] = tf[i];
                }
                if (local_tf[i] > local_tf_[local_vocab.vocabs_[i]])
                {
                    local_tf_[local_vocab.vocabs_[i]] = local_tf[i];
                }
            }
        }

        delete[] local_tf;
        delete[] tf;

        if(!Config::inference)
        {
            ModelSchedule();
        }
        else
        {
            ModelSchedule4Inference();
        }
        BuildAliasIndex();
    }

    void Meta::ModelSchedule()
    {
        int64_t model_capacity = Config::model_capacity;
        int64_t alias_capacity = Config::alias_capacity;
        int64_t delta_capacity = Config::delta_capacity;

        int32_t model_thresh = Config::num_topics / (2 * kLoadFactor);
        int32_t alias_thresh = (Config::num_topics * 2) / 3;
        int32_t delta_thresh = Config::num_topics / (4 * kLoadFactor);


		// Schedule for each data block
        for (int32_t i = 0; i < Config::num_blocks; ++i)
		{
			LocalVocab& local_vocab = local_vocabs_[i];
            int32_t* vocabs = local_vocab.vocabs_;
			local_vocab.slice_index_.push_back(0);

            int64_t model_offset = 0;
            int64_t alias_offset = 0;
            int64_t delta_offset = 0;
            for (int32_t j = 0; j < local_vocab.size_; ++j)
			{
                int32_t word = vocabs[j];
                int32_t tf = tf_[word];
                int32_t local_tf = local_tf_[word];
                int32_t model_size = (tf > model_thresh) ?
                    Config::num_topics* sizeof(int32_t) :
                    tf * kLoadFactor * sizeof(int32_t);
                model_offset += model_size;

                int32_t alias_size = (tf > alias_thresh) ?
                    Config::num_topics * 2 * sizeof(int32_t) :
                    tf * 3 * sizeof(int32_t);
                alias_offset += alias_size;

                int32_t delta_size = (local_tf > delta_thresh) ?
                    Config::num_topics * sizeof(int32_t) :
                    local_tf * kLoadFactor * 2 * sizeof(int32_t);
                delta_offset += delta_size;

                if (model_offset > model_capacity ||
                    alias_offset > alias_capacity ||
                    delta_offset > delta_capacity)
                {
                    Log::Info("Actual Model capacity: %d MB, Alias capacity: %d MB, Delta capacity: %dMB\n",
                        model_offset/1024/1024, alias_offset/1024/1024, delta_offset/1024/1024);
                    Log::Info("Actual asymmetric alpha capacity: %d MB, Alias capacity: %dMB, Delta capacity: %d MB\n",
                        Config::num_topics * (kMaxDocLength + 1) * sizeof(int32_t)/1024/1024,
                        2 * Config::num_topics * sizeof(int32_t)/1024/1024,
                        Config::num_topics * (kMaxDocLength + 1) * sizeof(int32_t)/1024/1024);
                    local_vocab.slice_index_.push_back(j);
                    ++local_vocab.num_slices_;
                    model_offset = model_size;
                    alias_offset = alias_size;
                    delta_offset = delta_size;
                }
			}
            local_vocab.slice_index_.push_back(local_vocab.size_);
            ++local_vocab.num_slices_;
            Log::Info("INFO: block = %d, the number of slice = %d\n",
                i, local_vocab.num_slices_);
		}
    }

    void Meta::ModelSchedule4Inference()
    {
        Config::alias_capacity = 0;
        int32_t alias_thresh = (Config::num_topics * 2) / 3;
        // Schedule for each data block
        for (int32_t i = 0; i < Config::num_blocks; ++i)
        {
            LocalVocab& local_vocab = local_vocabs_[i];
            int32_t* vocabs = local_vocab.vocabs_;
            local_vocab.slice_index_.push_back(0);
            local_vocab.slice_index_.push_back(local_vocab.size_);
            local_vocab.num_slices_ = 1;
            int64_t alias_offset = 0;
            for (int32_t j = 0; j < local_vocab.size_; ++j)
            {
                int32_t word = vocabs[j];
                int32_t tf = tf_[word];
                int32_t alias_size = (tf > alias_thresh) ?
                    Config::num_topics * 2 * sizeof(int32_t) :
                    tf * 3 * sizeof(int32_t);
                alias_offset += alias_size;
            }
            if(alias_offset > Config::alias_capacity)
            {
                Config::alias_capacity = alias_offset;
            }
        }
        Log::Info("Actual Alias capacity: %d MB\n", Config::alias_capacity/1024/1024);
    }

    void Meta::BuildAliasIndex()
    {
        int32_t alias_thresh = (Config::num_topics * 2) / 3;
        alias_index_.resize(Config::num_blocks);
        // for each block
        for (int32_t i = 0; i < Config::num_blocks; ++i)
        {
            const LocalVocab& vocab = local_vocab(i);
            alias_index_[i].resize(vocab.num_slice());
            // for each slice
            for (int32_t j = 0; j < vocab.num_slice(); ++j)
            {
                alias_index_[i][j] = new AliasTableIndex();
                int64_t offset = 0;
                for (const int32_t* p = vocab.begin(j);
                    p != vocab.end(j); ++p)
                {
                    int32_t word = *p;
                    bool is_dense = true;
                    int32_t capacity = Config::num_topics;
                    int64_t size = Config::num_topics * 2;
                    if (tf(word) < alias_thresh)
                    {
                        is_dense = false;
                        capacity = tf(word);
                        size = tf(word) * 3;
                    }
                    alias_index_[i][j]->PushWord(word, is_dense, offset, capacity);
                    offset += size;
                }
            }
        }
    }

} // namespace lightlda
} // namespace multiverso
