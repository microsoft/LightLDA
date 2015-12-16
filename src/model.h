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
    Model(Meta * meta): meta_(meta) {}
    void Initialize();
  private:
    void CreateTable();
    void ConfigTable();
    void LoadTables();
    void LoadWordTopicTable(const std::string& model_fname);
    void LoadSummaryTable(const std::string& model_fname);
  private:
    Meta * meta_;
  };
} // namespace lightlda
} // namespace multiverso

#endif // LIGHTLDA_MODEL_H_
