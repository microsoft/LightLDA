#include "model.h"
#include "common.h"
#include "meta.h"
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <regex.h>
#include <algorithm>
#include <multiverso/log.h>

namespace multiverso { namespace lightlda
{
  Model::Model(Meta * meta): word_topic_table_(nullptr), 
    summary_row_(nullptr), meta_(meta)
  {
    CreateTable();
  }

  void Model::Init()
  {
    LoadTable();
  }

  void Model::CreateTable()
  {
    int32_t num_vocabs = Config::num_vocabs;
    int32_t num_topics = Config::num_topics;
    multiverso::Format dense_format = multiverso::Format::Dense;
    multiverso::Format sparse_format = multiverso::Format::Sparse;
    Type int_type = Type::Int;
    Type longlong_type = Type::LongLong;

    word_topic_table_.reset(new Table(kWordTopicTable, num_vocabs, num_topics,
            int_type, dense_format));
    summary_row_.reset(new Table(kSummaryRow, 1, num_topics,
            longlong_type, dense_format));
  }

  void Model::LoadTable()
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
    multiverso::Format dense_format = multiverso::Format::Dense;
    multiverso::Format sparse_format = multiverso::Format::Sparse;
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
      if(meta_->tf(word_id) > 0)
      {
        //set row
        if (meta_->tf(word_id) * kLoadFactor > Config::num_topics)
        {
          word_topic_table_->SetRow(word_id, dense_format, Config::num_topics);
        }
        else
        {
          word_topic_table_->SetRow(word_id, sparse_format, meta_->tf(word_id) * kLoadFactor);
        }
        //get row
        Row<int32_t> * row = static_cast<Row<int32_t>*>
          (word_topic_table_->GetRow(word_id));

        //add features to row
        while (ss >> fea)
        {
          size_t pos = fea.find_last_of(':');
          if(pos != std::string::npos)
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

  void Model::LoadSummaryTable(const std::string& model_fname)
  {
    Row<int64_t> * row = static_cast<Row<int64_t>*>
      (summary_row_->GetRow(0));
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
} // namespace lightlda
} // namespace multiverso
