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
    static void Initialize(Meta * meta);
  private:
    static void CreateTable();
    static void ConfigTable(Meta * meta);
    static void LoadTables();
    static void LoadWordTopicTable(const std::string& model_fname);
    static void LoadSummaryTable(const std::string& model_fname);

  };
} // namespace lightlda
} // namespace multiverso

#endif // LIGHTLDA_MODEL_H_
