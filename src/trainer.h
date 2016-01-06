/*!
 * \file trainer.h
 * \brief Defines multiverso interface for parameter loading and data training
 */

#ifndef LIGHTLDA_TRAINER_H_
#define LIGHTLDA_TRAINER_H_

#include <mutex>

#include <multiverso/multiverso.h>
#include <multiverso/barrier.h>

namespace multiverso { namespace lightlda
{
    class AliasTable;
    class LDADataBlock;
    class LightDocSampler;
    class Meta;
    class PSModel;

    /*! \brief Trainer is responsible for training a data block */
    class Trainer : public TrainerBase
    {
    public:
        Trainer(AliasTable* alias, Barrier* barrier, Meta* meta);
        ~Trainer();
        /*!
         * \brief Defines Trainning method for a data_block in one iteration
         * \param data_block pointer to data block base
         */
        void TrainIteration(DataBlockBase* data_block) override;
        /*!
         * \brief Evaluates a data block, compute its loss function
         * \param block pointer to data block
         */
        void Evaluate(LDADataBlock* block);

        void Dump(int32_t iter, LDADataBlock* lda_data_block);

    private:
        /*! \brief alias table, for alias access */
        AliasTable* alias_;
        /*! \brief sampler for lightlda */
        LightDocSampler* sampler_;
        /*! \brief barrier for thread-sync */
        Barrier* barrier_;
        /*! \brief meta information */
        Meta* meta_;
        /*! \brief model acceccor */
        PSModel * model_;
        static std::mutex mutex_;

        static double doc_llh_;
        static double word_llh_;
    };

    /*! 
     * \brief ParamLoader is responsible for parsing a data block and
     *        preload parameters needed by this block
     */
    class ParamLoader : public ParameterLoaderBase
    {
        /*!
         * \brief Parse a data block to record which parameters (word) is 
         *        needed for training this block
         */
        void ParseAndRequest(DataBlockBase* data_block) override;
    };

} // namespace lightlda
} // namespace multiverso

#endif // LIGHTLDA_TRAINER_H_
