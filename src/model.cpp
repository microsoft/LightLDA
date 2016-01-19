#include "model.h"

#ifdef _MSC_VER
// TODO
#else
#include <dirent.h>
#include <regex.h>
#endif

#include <algorithm>
#include <fstream>
#include <sstream>

#include "meta.h"
#include "trainer.h"
#include "data_stream.h"
#include "data_block.h"
#include "document.h"

#include <multiverso/log.h>
#include <multiverso/multiverso.h>

namespace multiverso { namespace lightlda
{
    // -- local model implement area --------------------------------- //
    LocalModel::LocalModel(Meta * meta) : 
        word_topic_table_(nullptr), summary_table_(nullptr), 
        topic_frequency_table_(nullptr), doc_length_table_(nullptr),
        meta_(meta)
    {
        CreateTable();
    }

    void LocalModel::Init()
    {
        LoadTable();
    }

    void LocalModel::CreateTable()
    {
        int32_t num_vocabs = Config::num_vocabs;
        int32_t num_topics = Config::num_topics;
        multiverso::Format dense_format = multiverso::Format::Dense;
        //multiverso::Format sparse_format = multiverso::Format::Sparse;
        Type int_type = Type::Int;
        Type longlong_type = Type::LongLong;

        word_topic_table_.reset(new Table(kWordTopicTable, num_vocabs, num_topics,
            int_type, dense_format));
        summary_table_.reset(new Table(kSummaryRow, 1, num_topics,
            longlong_type, dense_format));
    }

    void LocalModel::LoadTable()
    {
#ifdef _MSC_VER
        Log::Fatal("Not implementent yet on Windows\n");
#else
        Log::Info("loading model\n");
        //set regex for model files
        regex_t model_wordtopic_regex;
        regex_t model_summary_regex;
        std::string prefix = "server_[[:digit:]]+_table_";
        std::string suffix = ".model";
        std::ostringstream wordtopic_regstr;
        wordtopic_regstr << prefix << kWordTopicTable << suffix;
        std::ostringstream summary_regstr;
        summary_regstr << prefix << kSummaryRow << suffix;
        regcomp(&model_wordtopic_regex, wordtopic_regstr.str().c_str(), REG_EXTENDED);
        regcomp(&model_summary_regex, summary_regstr.str().c_str(), REG_EXTENDED);

        //look for model files & load them
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir(Config::input_dir.c_str())) != NULL)
        {
            while ((ent = readdir(dir)) != NULL)
            {
                if (!regexec(&model_wordtopic_regex, ent->d_name, 0, NULL, 0))
                {
                    Log::Info("loading word topic table[%s]\n", ent->d_name);
                    LoadWordTopicTable(Config::input_dir + "/" + ent->d_name);
                }
                else if (!regexec(&model_summary_regex, ent->d_name, 0, NULL, 0))
                {
                    Log::Info("loading summary table[%s]\n", ent->d_name);
                    LoadSummaryTable(Config::input_dir + "/" + ent->d_name);
                }
            }
            closedir(dir);
        }
        else
        {
            Log::Fatal("model dir does not exist : %s\n", Config::input_dir.c_str());
        }
        regfree(&model_wordtopic_regex);
        regfree(&model_summary_regex);
#endif
    }

    void LocalModel::LoadWordTopicTable(const std::string& model_fname)
    {
        multiverso::Format dense_format = multiverso::Format::Dense;
        multiverso::Format sparse_format = multiverso::Format::Sparse;
        std::ifstream model_file(model_fname, std::ios::in);
        std::string line;
        while (getline(model_file, line))
        {
            std::stringstream ss(line);
            std::string word;
            std::string fea;
            std::vector<std::string> feas;
            int32_t word_id, topic_id, freq;
            //assign word id
            ss >> word;
            word_id = std::stoi(word);
            if (meta_->tf(word_id) > 0)
            {
                //set row
                if (meta_->tf(word_id) * kLoadFactor > Config::num_topics)
                {
                    word_topic_table_->SetRow(word_id, dense_format, 
                        Config::num_topics);
                }
                else
                {
                    word_topic_table_->SetRow(word_id, sparse_format, 
                        meta_->tf(word_id) * kLoadFactor);
                }
                //get row
                Row<int32_t> * row = static_cast<Row<int32_t>*>
                    (word_topic_table_->GetRow(word_id));

                //add features to row
                while (ss >> fea)
                {
                    size_t pos = fea.find_last_of(':');
                    if (pos != std::string::npos)
                    {
                        topic_id = std::stoi(fea.substr(0, pos));
                        freq = std::stoi(fea.substr(pos + 1));
                        row->Add(topic_id, freq);
                    }
                    else
                    {
                        Log::Fatal("bad format of model: %s\n", line.c_str());
                    }
                }
            }
        }
        model_file.close();
    }

    void LocalModel::LoadSummaryTable(const std::string& model_fname)
    {
        Row<int64_t> * row = static_cast<Row<int64_t>*>
            (summary_table_->GetRow(0));
        std::ifstream model_file(model_fname, std::ios::in);
        std::string line;
        if (getline(model_file, line))
        {
            std::stringstream ss(line);
            std::string fea;
            std::vector<std::string> feas;
            int32_t topic_id, freq;
            //skip word id
            ss >> fea;
            //add features to row
            while (ss >> fea)
            {
                size_t pos = fea.find_last_of(':');
                if (pos != std::string::npos)
                {
                    topic_id = std::stoi(fea.substr(0, pos));
                    freq = std::stoi(fea.substr(pos + 1));
                    row->Add(topic_id, freq);
                }
                else
                {
                    Log::Fatal("bad format of model: %s\n", line.c_str());
                }
            }
        }
        model_file.close();
    }

    void LocalModel::AddWordTopic(
        integer_t word_id, integer_t topic_id, int32_t delta) 
    {
        Log::Fatal("Not implemented yet\n");
    }

    void LocalModel::AddSummary(integer_t topic_id, int64_t delta) 
    {
        Log::Fatal("Not implemented yet\n");
    }

    Row<int32_t>& LocalModel::GetWordTopicRow(integer_t word)
    {
        return *(static_cast<Row<int32_t>*>(word_topic_table_->GetRow(word)));
    }

    Row<int64_t>& LocalModel::GetSummaryRow()
    {
        return *(static_cast<Row<int64_t>*>(summary_table_->GetRow(0)));
    }

    Row<int32_t>& LocalModel::GetTopicFrequencyRow(integer_t topic_id)
    {
        return *(static_cast<Row<int32_t>*>(topic_frequency_table_->GetRow(topic_id)));
    }
    Row<int32_t>& LocalModel::GetDocLengthRow()
    {
        return *(static_cast<Row<int32_t>*>(doc_length_table_->GetRow(0)));
    }
    void LocalModel::AddTopicFrequency(integer_t topic_id, integer_t freq, 
        int32_t delta)
    {
        Log::Fatal("Not implemented yet\n");
    }
    void LocalModel::AddDocLength(integer_t doc_len, int32_t delta)
    {
        Log::Fatal("Not implemented yet\n");
    }
    // -- local model implement area --------------------------------- //

    // -- PS model implement area --------------------------------- //
    void PSModel::Init(Meta* meta, IDataStream * data_stream)
    {
        Multiverso::BeginConfig();
        CreateTable();
        ConfigTable(meta);
        LoadTable(meta, data_stream);
        Multiverso::EndConfig();
    }

    void PSModel::CreateTable()
    {
        int32_t num_vocabs = Config::num_vocabs;
        int32_t num_topics = Config::num_topics;
        Type int_type = Type::Int;
        Type longlong_type = Type::LongLong;
        multiverso::Format dense_format = multiverso::Format::Dense;
        //multiverso::Format sparse_format = multiverso::Format::Sparse;

        Multiverso::AddServerTable(kWordTopicTable, num_vocabs,
            num_topics, int_type, dense_format);
        Multiverso::AddCacheTable(kWordTopicTable, num_vocabs,
            num_topics, int_type, dense_format, Config::model_capacity);
        Multiverso::AddAggregatorTable(kWordTopicTable, num_vocabs,
            num_topics, int_type, dense_format, Config::delta_capacity);    
        Multiverso::AddTable(kSummaryRow, 1, Config::num_topics,
            longlong_type, dense_format);

        if(Config::asymmetric_prior)
        {
            Multiverso::AddServerTable(kTopicFrequencyTable, num_topics,
                kMaxDocLength, int_type, dense_format);
            Multiverso::AddCacheTable(kTopicFrequencyTable, num_topics,
                kMaxDocLength, int_type, dense_format, 
                num_topics * kMaxDocLength * sizeof(int32_t));
            Multiverso::AddAggregatorTable(kTopicFrequencyTable, num_vocabs,
                num_topics, int_type, dense_format,
                num_topics * kMaxDocLength * sizeof(int32_t)); 
            Multiverso::AddTable(kDocLengthRow, 1, kMaxDocLength,
                int_type, dense_format);
        }
    }

    void PSModel::ConfigTable(Meta* meta)
    {
        multiverso::Format dense_format = multiverso::Format::Dense;
        multiverso::Format sparse_format = multiverso::Format::Sparse;
        for (int32_t word = 0; word < Config::num_vocabs; ++word)
        {
            if (meta->tf(word) > 0)
            {
                if (meta->tf(word) * kLoadFactor > Config::num_topics)
                {
                    Multiverso::SetServerRow(kWordTopicTable,
                        word, dense_format, Config::num_topics);
                    Multiverso::SetCacheRow(kWordTopicTable,
                        word, dense_format, Config::num_topics);
                }
                else
                {
                    Multiverso::SetServerRow(kWordTopicTable,
                        word, sparse_format, meta->tf(word) * kLoadFactor);
                    Multiverso::SetCacheRow(kWordTopicTable,
                        word, sparse_format, meta->tf(word) * kLoadFactor);
                }
            }
            if (meta->local_tf(word) > 0)
            {
                if (meta->local_tf(word) * 2 * kLoadFactor > Config::num_topics)
                    Multiverso::SetAggregatorRow(kWordTopicTable, 
                        word, dense_format, Config::num_topics);
                else
                    Multiverso::SetAggregatorRow(kWordTopicTable, word, 
                        sparse_format, meta->local_tf(word) * 2 * kLoadFactor);
            }
        }
        if(Config::asymmetric_prior)
        {
            for(int32_t topic = 0; topic < Config::num_topics; topic++)
            {
                Multiverso::SetServerRow(kTopicFrequencyTable,
                        topic, dense_format, kMaxDocLength);
                Multiverso::SetCacheRow(kTopicFrequencyTable,
                        topic, dense_format, kMaxDocLength);
                Multiverso::SetAggregatorRow(kTopicFrequencyTable,
                        topic, dense_format, kMaxDocLength);
            }
        }
    }
    
    void PSModel::LoadTable(Meta* meta, IDataStream * data_stream)
    {
        int32_t t, c;
        std::unique_ptr<Row<int32_t>> doc_topic_counter;
        doc_topic_counter.reset(new Row<int32_t>(0, 
            multiverso::Format::Sparse, kMaxDocLength));
        for (int32_t block = 0; block < Config::num_blocks; ++block)
        {
            data_stream->BeforeDataAccess();
            DataBlock& data_block = data_stream->CurrDataBlock();
            int32_t num_slice = meta->local_vocab(block).num_slice();
            for (int32_t slice = 0; slice < num_slice; ++slice)
            {
                for (int32_t i = 0; i < data_block.Size(); ++i)
                {
                    Document* doc = data_block.GetOneDoc(i);
                    int32_t& cursor = doc->Cursor();
                    if (slice == 0) cursor = 0;
                    int32_t last_word = meta->local_vocab(block).LastWord(slice);
                    // Init the server table
                    for (; cursor < doc->Size(); ++cursor)
                    {
                        if (doc->Word(cursor) > last_word) break;
                        Multiverso::AddToServer<int32_t>(kWordTopicTable,
                            doc->Word(cursor), doc->Topic(cursor), 1);
                        Multiverso::AddToServer<int64_t>(kSummaryRow,
                            0, doc->Topic(cursor), 1);
                    }
                    if(Config::asymmetric_prior && 0 == slice) 
                    {
                        doc_topic_counter->Clear();
                        doc->GetDocTopicVector(*doc_topic_counter);
                        Row<int32_t>::iterator iter = doc_topic_counter->Iterator();
                        while (iter.HasNext())
                        {
                            t = iter.Key();
                            c = iter.Value();
                            Multiverso::AddToServer<int32_t>(kTopicFrequencyTable,
                                t, c, 1);
                            iter.Next();
                        }
                        Multiverso::AddToServer<int32_t>(kDocLengthRow, 0, doc->Size(), 1);
                    }
                }
                Multiverso::Flush();
            }
            data_stream->EndDataAccess();
        }
    }

    Row<int32_t>& PSModel::GetWordTopicRow(integer_t word_id)
    {
        return trainer_->GetRow<int32_t>(kWordTopicTable, word_id);
    }

    Row<int64_t>& PSModel::GetSummaryRow()
    {
        return trainer_->GetRow<int64_t>(kSummaryRow, 0);
    }

    void PSModel::AddWordTopic(
        integer_t word_id, integer_t topic_id, int32_t delta)
    {
        trainer_->Add<int32_t>(kWordTopicTable, word_id, topic_id, delta);
    }

    void PSModel::AddSummary(integer_t topic_id, int64_t delta)
    {
        trainer_->Add<int64_t>(kSummaryRow, 0, topic_id, delta);
    }

    Row<int32_t>& PSModel::GetTopicFrequencyRow(integer_t topic_id)
    {
        return trainer_->GetRow<int32_t>(kTopicFrequencyTable, topic_id);
    }
    Row<int32_t>& PSModel::GetDocLengthRow()
    {
        return trainer_->GetRow<int32_t>(kDocLengthRow, 0);
    }
    void PSModel::AddTopicFrequency(integer_t topic_id, integer_t freq, 
        int32_t delta)
    {
        trainer_->Add<int32_t>(kTopicFrequencyTable, topic_id, freq, delta);
    }
    void PSModel::AddDocLength(integer_t doc_len, int32_t delta)
    {
        trainer_->Add<int32_t>(kSummaryRow, 0, doc_len, delta);
    }
    // -- PS model implement area --------------------------------- //

} // namespace lightlda
} // namespace multiverso
