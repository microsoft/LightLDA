#include "model.h"
#include "meta.h"
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <regex.h>
#include <multiverso/log.h>

namespace multiverso { namespace lightlda
{
  void Model::Initialize(Meta * meta)
  {
    CreateTable();
    ConfigTable(meta);
    LoadTables();
  }

  void Model::CreateTable()
  {
    int32_t num_vocabs = Config::num_vocabs;
    int32_t num_topics = Config::num_topics;
    Type int_type = Type::Int;
    Type longlong_type = Type::LongLong;
    multiverso::Format dense_format = multiverso::Format::Dense;
    Multiverso::AddServerTable(kWordTopicTable, num_vocabs,
        num_topics, int_type, dense_format);
    Multiverso::AddCacheTable(kWordTopicTable, num_vocabs,
        num_topics, int_type, dense_format, Config::model_capacity);
    Multiverso::AddAggregatorTable(kWordTopicTable, num_vocabs,
        num_topics, int_type, dense_format, Config::delta_capacity);
    Multiverso::AddTable(kSummaryRow, 1, Config::num_topics,
        longlong_type, dense_format);
  }

  void Model::ConfigTable(Meta * meta)
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
  }

  void Model::LoadTables()
  {
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
    if ((dir = opendir (Config::input_dir.c_str())) != NULL)
    {
      while ((ent = readdir (dir)) != NULL)
      {
        if(!regexec(&model_wordtopic_regex, ent->d_name, 0, NULL, 0))
        {
          Log::Info("loading word topic table[%s]\n", ent->d_name);
          LoadWordTopicTable(Config::input_dir + "/" + ent->d_name);
        }
        else if(!regexec(&model_summary_regex, ent->d_name, 0, NULL, 0))
        {
          Log::Info("loading summary table[%s]\n", ent->d_name);
          LoadSummaryTable(Config::input_dir + "/" + ent->d_name);
        }
      }
      closedir (dir);
    }
    else
    {
      Log::Fatal("model dir does not exist : %s\n", Config::input_dir.c_str());
    }
    regfree(&model_wordtopic_regex);
    regfree(&model_summary_regex);
  }

  void Model::LoadWordTopicTable(const std::string& model_fname)
  {
    std::ifstream model_file(model_fname, std::ios::in);
    std::string line;
    while(getline(model_file, line))
    {
      std::stringstream ss(line);
      std::string word;
      std::string fea;
      std::vector<std::string> feas;
      int32_t word_id, topic_id, freq;
      //assign word id
      ss >> word;
      word_id = std::stoi(word);
      //add features to row
      while (ss >> fea)
      {
        size_t pos = fea.find_last_of(':');
        if(pos != std::string::npos)
        {
          topic_id = std::stoi(fea.substr(0, pos));
          freq = std::stoi(fea.substr(pos + 1));
          Multiverso::AddToServer<int32_t>(kWordTopicTable, word_id, topic_id, freq);
        }
        else
        {
            Log::Fatal("bad format of model: %s\n", line.c_str());
        }
      }
    }
    model_file.close();
  }

  void Model::LoadSummaryTable(const std::string& model_fname)
  {
    std::ifstream model_file(model_fname, std::ios::in);
    std::string line;
    if(getline(model_file, line))
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
        if(pos != std::string::npos)
        {
          topic_id = std::stoi(fea.substr(0, pos));
          freq = std::stoi(fea.substr(pos + 1));
          Multiverso::AddToServer<int64_t>(kSummaryRow, 0, topic_id, freq);
        }
        else
        {
            Log::Fatal("bad format of model: %s\n", line.c_str());
        }
      }
    }
    model_file.close();
  }
} // namespace lightlda
} // namespace multiverso
