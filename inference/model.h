/*!
 * \file model.h
 * \brief define local model reader
 */


#ifndef LIGHTLDA_MODEL_H_
#define LIGHTLDA_MODEL_H_

#include "common.h"
#include <multiverso/multiverso.h>
#include <string>

namespace multiverso { namespace lightlda
{
  class Meta;

  class Model
  {

  public:
    Model(Meta * meta);

    void Init();
    
    RowBase* GetWordTopicRow(integer_t word_id)
    {
      return word_topic_table_->GetRow(word_id);
    }

    RowBase* GetSummaryRow(integer_t topic_id)
    {
      return summary_row_->GetRow(topic_id);
    }

  private:
    void CreateTable();
    void LoadTable();
    void LoadWordTopicTable(const std::string& model_fname);
    void LoadSummaryTable(const std::string& model_fname);

  private:
    std::unique_ptr<Table> word_topic_table_;
    std::unique_ptr<Table> summary_row_;
    Meta* meta_;
  };
} // namespace lightlda
} // namespace multiverso

#endif // LIGHTLDA_MODEL_H_
