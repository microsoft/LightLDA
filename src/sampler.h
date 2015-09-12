/*!
 * \file util.h
 * \brief Defines lightlda samplers
 */

#ifndef _LIGHTLDA_SAMPLER_H_
#define _LIGHTLDA_SAMPLER_H_

#include <memory>
#include "util.h"

namespace multiverso
{
    template <typename T>
    class Row;
}

namespace multiverso { namespace lightlda
{
    class AliasTable;
    class Document;
    class Trainer;
    
    /*! \brief lightlda sampler */
    class LightDocSampler
    {
    public:
        LightDocSampler();
        /*! 
         * \brief Sample one document, update latent topic assignment 
         *  and statistics
         * \param doc pointer to document
         * \param slice slice id
         * \param lastword last word of current slice
         * \param trainer pointer to trainer, for access of model
         * \param alias pointer to alias table, for access of alias
         * \return number of sampled token
         */
        int32_t SampleOneDoc(Document* doc, int32_t slice, int32_t lastword,
            Trainer* trainer, AliasTable* alias);
        /*!
         * \brief Get doc-topic-counter, for reusing this container
         * \return reference to light hash map
         */
        Row<int32_t>& doc_topic_counter() { return *doc_topic_counter_; }
    private:
        /*!
         * \brief Init document before sampling
         * \param doc pointer to document
         */
        void DocInit(Document* doc);
        /*!
         * \brief Sample the latent topic assignment for a token 
         * \param doc current document
         * \param word current token
         * \param state state of the word
         * \param old_topic old topic assignment of this token
         * \param trainer for model access
         * \param alias for alias table access
         */
        int32_t Sample(Document* doc, int32_t word, int32_t state, 
            int32_t old_topic, Trainer* trainer, AliasTable* alias);

        /*! 
         * \brief Sample the latent topic assignment for a token. This function
         *  make a little approximation to the proper Metropolis-Hasting 
         *  algorithm, but empirically this converges as good as exact Sample, 
         *  with faster speed.
         * \param same with Sample
         */
        int32_t ApproxSample(Document* doc, int32_t word, int32_t state, 
            int32_t old_topic, Trainer* trainer, AliasTable* alias);
    private:
        // lda hyper-parameter
        float alpha_;
        float beta_;
        float alpha_sum_;
        float beta_sum_;

        int32_t num_vocab_;
        int32_t num_topic_;
        int32_t mh_steps_;

        xorshift_rng rng_;
        std::unique_ptr<Row<int32_t>> doc_topic_counter_;
    };
} // namespace lightlda
} // namespace multiverso

#endif // _LIGHTLDA_SAMPLER_H_
