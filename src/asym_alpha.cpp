#include "asym_alpha.h"
#include "alias_table.h"
#include "model.h"
#include "common.h"
#include <multiverso/row.h>

namespace multiverso 
{    
namespace lightlda
{
    AsymAlpha::AsymAlpha() : dirichlet_scale_(1.0), dirichlet_shape_(1.00001)
    {
        num_topic_ = Config::num_topics;
        max_doc_length_ = kMaxDocLength;
        num_alpha_iterations_ = Config::num_alpha_iterations;
        alpha_sum_ = num_topic_ * Config::alpha;
        non_zero_limit_.resize(num_topic_);
        alpha_base_measure_.resize(num_topic_, Config::alpha);
        kv_vector_ = new int32_t[2 * num_topic_];
        alias_rng_int_ = new AliasMultinomialRNGInt(num_topic_);
    }

    AsymAlpha::~AsymAlpha()
    {
        delete [] kv_vector_;
        delete alias_rng_int_;
    }

    void AsymAlpha::LearnDirichletPrior(ModelBase * model)
    {
        float oldParametersK;
        float currentDigamma;
        float denominator;
        int nonZeroLimit;
        float parametersSum;
        
        // get the initial non_zero_limit_
        for (int k = 0; k < num_topic_; ++k)
        {
            non_zero_limit_[k] = 0;
            Row<int32_t>& row = model->GetTopicFrequencyRow(k);
            for (int i = 1; i <= max_doc_length_; ++i)
            {
                if (row.At(i) > 0)
                {
                    non_zero_limit_[k] = i;
                }
            }
        }

        // get the initial atomic_alpha_sum_
        parametersSum = 0;
        for (int k = 0; k < num_topic_; k++)
        {
            parametersSum += alpha_base_measure_[k];
        }

        Row<int32_t>& doc_length_row = model->GetDocLengthRow();

        for (int iteration = 0;
            iteration < num_alpha_iterations_; ++iteration)
        {
            // Calculate the denominator
            denominator = 0;
            currentDigamma = 0;

            // Iterate over the histogram:
            for (int i = 1; i <= max_doc_length_; i++)
            {
                currentDigamma += 1 / (parametersSum + i - 1);
                denominator += doc_length_row.At(i) * currentDigamma;
            }
            // Bayesian estimation Part I
            denominator -= 1 / dirichlet_scale_;

            // Calculate the individual parameters
            parametersSum = 0;

            for (int k = 0; k < num_topic_; k++)
            {
                // What's the largest non-zero element in the histogram?
                nonZeroLimit = non_zero_limit_[k];

                oldParametersK = alpha_base_measure_[k];
                alpha_base_measure_[k] = 0;
                currentDigamma = 0;

                Row<int32_t>& row = model->GetTopicFrequencyRow(k);

                for (int i = 1; i <= nonZeroLimit; i++)
                {
                    currentDigamma += 1 / (oldParametersK + i - 1);
                    alpha_base_measure_[k] += row.At(i) * currentDigamma;
                }

                // Bayesian estimation part II
                alpha_base_measure_[k] = oldParametersK
                    * (alpha_base_measure_[k] + dirichlet_shape_)
                    / denominator;
                parametersSum += alpha_base_measure_[k];
            }
        }
        alpha_sum_ = parametersSum;
    }

    void AsymAlpha::BuildAlias()
    {
        alias_rng_int_->Build(alpha_base_measure_, num_topic_, 
            alpha_sum_, alpha_height_, kv_vector_);
    }

    int32_t AsymAlpha::Next()
    {
        return alias_rng_int_->Propose(rng_, alpha_height_, kv_vector_);
    }
} // namespace lightlda
} // namespace multiverso
