/*!
 * \file asym_alpha.h
 * \brief Defines asymmetric prior alpha
 */

#ifndef LIGHTLDA_ASYM_ALPHA_H_
#define LIGHTLDA_ASYM_ALPHA_H_

#include <vector>
#include <cstdint>
#include "util.h"

namespace multiverso 
{    
namespace lightlda
{
    class ModelBase;
    class AliasMultinomialRNGInt;

    class AsymAlpha
    {
    public:
        AsymAlpha();
        ~AsymAlpha();
        void LearnDirichletPrior(ModelBase * model);
        void BuildAlias();
        int32_t Next();
        float At(int32_t idx) const;
        float AlphaSum() const;
    private:
        int32_t num_topic_;
        int32_t max_doc_length_;
        int32_t num_alpha_iterations_;
        float dirichlet_scale_;
        float dirichlet_shape_;
        float alpha_sum_;
        int32_t alpha_height_;
        int32_t* kv_vector_;
        std::vector<int32_t> non_zero_limit_;
        std::vector<float> alpha_base_measure_;
        xorshift_rng rng_;
        AliasMultinomialRNGInt * alias_rng_int_;
    };

    // -- inline functions definition area --------------------------------- //
    inline float AsymAlpha::At(int32_t idx) const
    {
        return alpha_base_measure_[idx];
    }

    inline float AsymAlpha::AlphaSum() const
    {
        return alpha_sum_;
    }
    // -- inline functions definition area --------------------------------- //
} // namespace lightlda
} // namespace multiverso

#endif //LIGHTLDA_ASYM_ALPHA_H_
