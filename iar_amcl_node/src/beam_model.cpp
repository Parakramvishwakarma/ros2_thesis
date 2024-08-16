#include <math.h>
#include <assert.h>

#include "iar_amcl_node/beam_model.hpp"

namespace iar_amcl
{
    BeamModel::BeamModel(
    double z_hit, double z_short, double z_max, double z_rand, double sigma_hit,
    double lambda_short, size_t max_beams, map_t * map)
    : Laser(max_beams, map)
    {
        z_hit_ = z_hit;
        z_rand_ = z_rand;
        sigma_hit_ = sigma_hit;
        z_short_ = z_short;
        z_max_ = z_max;
        lambda_short_ = lambda_short;
    }

    // Determine the probability for the given pose
    double
    BeamModel::sensorFunction(nav2_amcl::LaserData * data, pf_sample_set_t * set)
    {
        BeamModel * self;
        int i, j, step;
        double z, pz;
        double p;
        double map_range;
        double obs_range, obs_bearing;
        double total_weight;
        pf_sample_t * sample;
        pf_vector_t pose;

        self = reinterpret_cast<BeamModel *>(data->laser);

        total_weight = 0.0;

        // Compute the sample weights
        for (j = 0; j < set->sample_count; j++) {
            sample = set->samples + j;
            pose = sample->pose;

            // Take account of the laser pose relative to the robot
            pose = pf_vector_coord_add(self->laser_pose_, pose);

            p = 1.0;

            step = (data->range_count - 1) / (self->max_beams_ - 1);
            for (i = 0; i < data->range_count; i += step) {
            obs_range = data->ranges[i][0];

            // Check for NaN
            if (isnan(obs_range)) {
                continue;
            }

            obs_bearing = data->ranges[i][1];

            // Compute the range according to the map
            map_range = map_calc_range(
                self->map_, pose.v[0], pose.v[1],
                pose.v[2] + obs_bearing, data->range_max);
            pz = 0.0;

            // Part 1: good, but noisy, hit
            z = obs_range - map_range;
            pz += self->z_hit_ * exp(-(z * z) / (2 * self->sigma_hit_ * self->sigma_hit_));

            // Part 2: short reading from unexpected obstacle (e.g., a person)
            if (z < 0) {
                pz += self->z_short_ * self->lambda_short_ * exp(-self->lambda_short_ * obs_range);
            }

            // Part 3: Failure to detect obstacle, reported as max-range
            if (obs_range == data->range_max) {
                pz += self->z_max_ * 1.0;
            }

            // Part 4: Random measurements
            if (obs_range < data->range_max) {
                pz += self->z_rand_ * 1.0 / data->range_max;
            }

            assert(pz <= 1.0);
            assert(pz >= 0.0);
            //      p *= pz;
            // here we have an ad-hoc weighting scheme for combining beam probs
            // works well, though...
            p += pz * pz * pz;
            }

            sample->weight *= p;
            total_weight += sample->weight;
        }

        return total_weight;
    }

    bool
    BeamModel::sensorUpdate(pf_t * pf, nav2_amcl::LaserData * data)
    {
        if (max_beams_ < 2) {
            return false;
        }
        // pf_update_sensor(pf, (pf_sensor_model_fn_t) sensorFunction, data);

        int i;
        pf_sample_set_t * set;
        pf_sample_t * sample;
        double total;

        set = pf->sets + pf->current_set;

        // Compute the sample weights
        total = sensorFunction(data, set);

        if (total > 0.0) {
            // Normalize weights
            double w_avg = 0.0;
            for (i = 0; i < set->sample_count; i++) {
                sample = set->samples + i;
                w_avg += sample->weight;
                sample->weight /= total;
            }
            // Update running averages of likelihood of samples (Prob Rob p258)
            w_avg /= set->sample_count;
            if (pf->w_slow == 0.0) {
                pf->w_slow = w_avg;
            } else {
                pf->w_slow += pf->alpha_slow * (w_avg - pf->w_slow);
            }
            if (pf->w_fast == 0.0) {
                pf->w_fast = w_avg;
            } else {
                pf->w_fast += pf->alpha_fast * (w_avg - pf->w_fast);
            }
        } else {
            // Handle zero total
            for (i = 0; i < set->sample_count; i++) {
                sample = set->samples + i;
                sample->weight = 1.0 / set->sample_count;
            }
        }

        return true;
    }
}