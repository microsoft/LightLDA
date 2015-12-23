/*!
 * \file inferer.h
 * \brief data inference
 */
#ifndef LIGHTLDA_INFERER_H_
#define LIGHTLDA_INFERER_H_

#include "trainer.h"
#include "common.h"
#include "model.h"
#include <pthread.h>
#include <multiverso/log.h>

namespace multiverso { namespace lightlda
{
    class AliasTable;
    class LDADataBlock;
    class LightDocSampler;
    class Meta;
    
    class Inferer: public Trainer
    {
    public:
        Inferer(AliasTable* alias_table, Meta* meta, Model * model,
                pthread_barrier_t* barrier, 
                int32_t id, int32_t thread_num):
            Trainer(alias_table, nullptr, meta), model_(model),
            barrier_(barrier), id_(id), thread_num_(thread_num) {}

        void TrainIteration(DataBlockBase* data_block) override;

        /*! \brief interface for accessing to model */
        Row<int32_t>& GetWordTopicRow(integer_t word_id) override;
        void UpdateWordTopic(integer_t word_id, integer_t topic_id, int32_t delta) override;
        Row<int64_t>& GetSummaryRow() override;
        void UpdateSummary(integer_t topic_id, int64_t delta) override;
    
    private:
        Model * model_;
        pthread_barrier_t* barrier_;
        int32_t id_;
        int32_t thread_num_;
    };
} // namespace lightlda
} // namespace multiverso


 #endif //LIGHTLDA_INFERER_H_