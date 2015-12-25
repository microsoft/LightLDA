/*!
 * \file model.h
 * \brief define local model reader
 */


#ifndef LIGHTLDA_MODEL_H_
#define LIGHTLDA_MODEL_H_

#include "trainer.h"
#include "common.h"
#include <multiverso/multiverso.h>
#include <string>

namespace multiverso { namespace lightlda
{
  class Meta;
  class Trainer;

  /*! \brief interface for acceess to model */
  class ModelBase
  {
  public:
      virtual ~ModelBase() {}
      virtual Row<int32_t>& GetWordTopicRow(integer_t word_id) = 0;
      virtual Row<int64_t>& GetSummaryRow() = 0;
      virtual void AddWordTopicRow(integer_t word_id, integer_t topic_id, int32_t delta) = 0;
      virtual void AddSummaryRow(integer_t topic_id, int64_t delta) = 0;
  };

  /*! \brief model based on local buffer */
  class LocalModel: public ModelBase
  {
  public:
    LocalModel(Meta * meta);
    void Init();

  public:
    Row<int32_t>& GetWordTopicRow(integer_t word_id) override;
    Row<int64_t>& GetSummaryRow() override;
    void AddWordTopicRow(integer_t word_id, integer_t topic_id, int32_t delta) override;
    void AddSummaryRow(integer_t topic_id, int64_t delta) override;

  private:
    void CreateTable();
    void LoadTable();
    void LoadWordTopicTable(const std::string& model_fname);
    void LoadSummaryTable(const std::string& model_fname);

  private:
    std::unique_ptr<Table> word_topic_table_;
    std::unique_ptr<Table> summary_table_;
    Meta* meta_;
  };

  /*! \brief model based on parameter server */
  class PSModel: public ModelBase
  {
  public:
    PSModel(Trainer* trainer): trainer_(trainer) {}
  public:
    Row<int32_t>& GetWordTopicRow(integer_t word_id) override;
    Row<int64_t>& GetSummaryRow() override;
    void AddWordTopicRow(integer_t word_id, integer_t topic_id, int32_t delta) override;
    void AddSummaryRow(integer_t topic_id, int64_t delta) override;
  private:
    Trainer* trainer_;
  };

  // -- inline functions definition area --------------------------------- //
  inline Row<int32_t>& LocalModel::GetWordTopicRow(integer_t word_id)
  {
      return *(static_cast<Row<int32_t>*>(word_topic_table_->GetRow(word_id)));
  }

  inline Row<int64_t>& LocalModel::GetSummaryRow()
  {
      return *(static_cast<Row<int64_t>*>(summary_table_->GetRow(0)));
  }

  inline void LocalModel::AddWordTopicRow(integer_t word_id, integer_t topic_id, int32_t delta) {}

  inline void LocalModel::AddSummaryRow(integer_t topic_id, int64_t delta) {}

  inline Row<int32_t>& PSModel::GetWordTopicRow(integer_t word_id)
  {
      return trainer_->GetRow<int32_t>(kWordTopicTable, word_id);
  }

  inline Row<int64_t>& PSModel::GetSummaryRow()
  {
      return trainer_->GetRow<int64_t>(kSummaryRow, 0);
  }

  inline void PSModel::AddWordTopicRow(integer_t word_id, integer_t topic_id, int32_t delta)
  {
      trainer_->Add<int32_t>(kWordTopicTable, word_id, topic_id, delta); 
  }

  inline void PSModel::AddSummaryRow(integer_t topic_id, int64_t delta)
  {
      trainer_->Add<int64_t>(kSummaryRow, 0, topic_id, delta);
  }
  // -- inline functions definition area --------------------------------- //

} // namespace lightlda
} // namespace multiverso

#endif // LIGHTLDA_MODEL_H_
