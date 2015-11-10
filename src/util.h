/*!
 * \file util.h
 * \brief Defines random number generator
 */

#ifndef LIGHTLDA_UTIL_H_
#define LIGHTLDA_UTIL_H_

#include <ctime>

namespace multiverso { namespace lightlda
{
    /*! \brief xorshift_rng is a random number generator */
    class xorshift_rng
    {
    public:
        xorshift_rng()
        {
            jxr_ = static_cast<unsigned int>(time(nullptr));
        }
        ~xorshift_rng() {}

        /*! \brief get random (xorshift) 32-bit integer*/
        int32_t rand()
        {
            jxr_ ^= (jxr_ << 13); jxr_ ^= (jxr_ >> 17); jxr_ ^= (jxr_ << 5);
            return jxr_ & 0x7fffffff;
        }

        double rand_double()
        {
            return rand() * 4.6566125e-10;
        }
        int32_t rand_k(int K)
        {
            return static_cast<int>(rand() * 4.6566125e-10 * K);
        }
    private:
        // No copying allowed
        xorshift_rng(const xorshift_rng &other);
        void operator=(const xorshift_rng &other);
        /*! \brief seed */
        uint32_t jxr_;
    };
} // namespace lightlda
} // namespace multiverso

#endif // LIGHTLDA_UTIL_H_
