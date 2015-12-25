/*!
 * \file inferer.h
 * \brief data inference
 */
#ifndef LIGHTLDA_INFERER_H_
#define LIGHTLDA_INFERER_H_

#include <pthread.h>
#include <multiverso/multiverso.h>
#include <multiverso/log.h>

namespace multiverso { namespace lightlda
{
    class AliasTable;
    class LDADataBlock;
    class LightDocSampler;
    class Meta;
    class LocalModel;
    
    class Inferer
    {
    public:
        Inferer(AliasTable* alias_table, Meta* meta, LocalModel * model,
                pthread_barrier_t* barrier, 
                int32_t id, int32_t thread_num);

        ~Inferer();

        void InferenceIteration(DataBlockBase* data_block);
    
    private:
        AliasTable* alias_;
        Meta* meta_;
        LocalModel * model_;
        pthread_barrier_t* barrier_;
        int32_t id_;
        int32_t thread_num_;
        LightDocSampler* sampler_;
    };
} // namespace lightlda
} // namespace multiverso


 #endif //LIGHTLDA_INFERER_H_