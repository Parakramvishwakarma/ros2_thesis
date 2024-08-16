#ifndef IAR_DIFFDRIVE_MOTION_MODEL_HPP_
#define IAR_DIFFDRIVE_MOTION_MODEL_HPP_

#include "nav2_amcl/pf/pf.hpp"


namespace iar_amcl
{
    class DifferentialMotionModel
    {
    public:
        DifferentialMotionModel(double alpha1, double alpha2, double alpha3, double alpha4);
        void odometryUpdate(pf_t * pf, const pf_vector_t & pose, const pf_vector_t & delta);
    private:
        double alpha1_;
        double alpha2_;
        double alpha3_;
        double alpha4_;
    };
}

#endif 