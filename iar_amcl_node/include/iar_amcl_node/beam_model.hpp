#ifndef IAR_AMCL__BEAM_MODEL_NODE_HPP_
#define IAR_AMCL__BEAM_MODEL_NODE_HPP_

#include "nav2_amcl/sensors/laser/laser.hpp"

namespace iar_amcl
{
    class BeamModel : public nav2_amcl::Laser
    {
    public:
        /*
        * @brief BeamModel constructor
        */
        BeamModel(
        double z_hit, double z_short, double z_max, double z_rand, double sigma_hit,
        double lambda_short, size_t max_beams, map_t * map);

        /*
        * @brief Run a sensor update on laser
        * @param pf Particle filter to use
        * @param data Laser data to use
        * @return if it was succesful
        */
        bool sensorUpdate(pf_t * pf, nav2_amcl::LaserData * data);

    private:
        static double sensorFunction(nav2_amcl::LaserData * data, pf_sample_set_t * set);
        double z_short_;
        double z_max_;
        double lambda_short_;
    };
}

#endif //IAR_AMCL__BEAM_MODEL_NODE_HPP_