#include "model.h"

#ifdef _MSC_VER
#include <io.h>
#include <regex>
#else
#include <dirent.h>
#include <regex.h>
#endif

#include <algorithm>
#include <fstream>
#include <sstream>

#include "meta.h"
#include "trainer.h"

#include <multiverso/log.h>
#include <multiverso/multiverso.h>

namespace multiverso { namespace lightlda
{
    LocalModel::LocalModel(Meta * meta) : word_topic_table_(nullptr),
        summary_table_(nullptr), meta_(meta)
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
        multiverso::Format sparse_format = multiverso::Format::Sparse;
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
        Log::Info("loading model\n");
        //set regex for model files
        std::string prefix = "server_[[:digit:]]+_table_";
        std::string suffix = ".model";
        std::ostringstream wordtopic_regstr;
        wordtopic_regstr << prefix << kWordTopicTable << suffix;
        std::ostringstream summary_regstr;
        summary_regstr << prefix << kSummaryRow << suffix;
        std::regex model_wordtopic_regex(wordtopic_regstr.str());
        std::regex model_summary_regex(summary_regstr.str());

        //look for model files & load them
        intptr_t handle;
        _finddata_t fileinto;
        std::string input_dir = Config::input_dir;
        handle = _findfirst(input_dir.append("\\*").c_str(), &fileinto);
        if (handle != -1)
        {
            do
            {
                if (std::regex_match(fileinto.name, fileinto.name + std::strlen(fileinto.name), model_wordtopic_regex))
                {
                    Log::Info("loading word topic table[%s]\n", fileinto.name);
                    LoadWordTopicTable(Config::input_dir + "/" + fileinto.name);
                }
                else if (std::regex_match(fileinto.name, fileinto.name + std::strlen(fileinto.name), model_summary_regex))
                {
                    Log::Info("loading summary table[%s]\n", fileinto.name);
                    LoadSummaryTable(Config::input_dir + "/" + fileinto.name);
                }
            } while (!_findnext(handle, &fileinto));
        }
        else
        {
            Log::Fatal("model dir does not exist : %s\n", Config::input_dir.c_str());
        }
        _findclose(handle);
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

    void LocalModel::AddWordTopicRow(
        integer_t word_id, integer_t topic_id, int32_t delta) 
    {
        Log::Fatal("Not implemented yet\n");
    }

    void LocalModel::AddSummaryRow(integer_t topic_id, int64_t delta) 
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
    
    Row<int32_t>& PSModel::GetWordTopicRow(integer_t word_id)
    {
        return trainer_->GetRow<int32_t>(kWordTopicTable, word_id);
    }

    Row<int64_t>& PSModel::GetSummaryRow()
    {
        return trainer_->GetRow<int64_t>(kSummaryRow, 0);
    }

    void PSModel::AddWordTopicRow(
        integer_t word_id, integer_t topic_id, int32_t delta)
    {
        trainer_->Add<int32_t>(kWordTopicTable, word_id, topic_id, delta);
    }

    void PSModel::AddSummaryRow(integer_t topic_id, int64_t delta)
    {
        trainer_->Add<int64_t>(kSummaryRow, 0, topic_id, delta);
    }

} // namespace lightlda
} // namespace multiverso
