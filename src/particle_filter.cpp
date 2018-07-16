/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // create random engine generator
  default_random_engine gen;
  
  // create normal distributions
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
  
  // set number of particles
  num_particles = 100;
  particles.resize(num_particles);
  weights.resize(num_particles);
  
	// sample all particles
	for (int i = 0; i < num_particles; ++i)
  {
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
		particles[i].weight = 1.0;
  }

  is_initialized = true;

  return;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  // create random engine generator
	default_random_engine gen;
  
  // create normal distributions
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);
  
  // predict all particles
	for (int i = 0; i < num_particles; ++i)
  {
    double theta = particles[i].theta;
    double yawrate_dt = yaw_rate * delta_t;
    
    if (fabs(yaw_rate) > 0.0001)
    {
      particles[i].x += velocity / yaw_rate * (sin(theta + yawrate_dt) - sin(theta));
      particles[i].y += velocity / yaw_rate * (cos(theta) - cos(theta + yawrate_dt));
      particles[i].theta += yawrate_dt;
    }
    else
    {
			particles[i].x += velocity * cos(theta) * delta_t;
      particles[i].y += velocity * sin(theta) * delta_t;
    }
    
    // add noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
  return;
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.

  // for each observation ...
  for (unsigned int i = 0; i < observations.size(); i++)
  {
    // ... search prediction with minimum distance
    double dist_min = std::numeric_limits<double>::max();
    int id_min;

    for (unsigned int j = 0; j < predicted.size(); j++)
    {
      double dist_pred = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
      if (dist_pred < dist_min)
      {
        id_min = j;
        dist_min = dist_pred;
      }
    }
    // store prediction id with minimum distance
    observations[i].id = predicted[id_min].id;
  }
  return;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
	const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  // update weight of each particle
  for (unsigned int i = 0; i < particles.size(); i++)
  {
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;

    double particle_x = particles[i].x;
    double particle_y = particles[i].y;
    double particle_theta = particles[i].theta;

    // get all landmarks, which are inside sensor range with respect to current particle
    vector<LandmarkObs> predictions;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++)
    {
      float landmark_x = map_landmarks.landmark_list[j].x_f;
      float landmark_y = map_landmarks.landmark_list[j].y_f;
      int landmark_id = map_landmarks.landmark_list[j].id_i;
      
      if (dist(landmark_x, landmark_y, particle_x, particle_y) <= sensor_range)
      {
        predictions.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
      }
    }

    // calculate global observations
    vector<LandmarkObs> global_observations;
    for (unsigned int j = 0; j < observations.size(); j++)
    {
      double global_observation_x = cos(particle_theta) * observations[j].x - sin(particle_theta) * observations[j].y + particle_x;
      double global_observation_y = sin(particle_theta) * observations[j].x + cos(particle_theta) * observations[j].y + particle_y;
      global_observations.push_back(LandmarkObs{observations[j].id, global_observation_x, global_observation_y});
    }

    // perform data association
    dataAssociation(predictions, global_observations);

    particles[i].weight = 1.0;
    for (unsigned int j = 0; j < global_observations.size(); j++)
    {
      double observation_x = global_observations[j].x;
      double observation_y = global_observations[j].y;      
      double prediction_x;
      double prediction_y;

      // search id of nearest observation
      for (unsigned int k = 0; k < predictions.size(); k++)
      {
        if (predictions[k].id == global_observations[j].id)
        {
          prediction_x = predictions[k].x;
          prediction_y = predictions[k].y;
        }
      }

      // collect associations
      associations.push_back(global_observations[j].id);
      sense_x.push_back(global_observations[j].x);
      sense_y.push_back(global_observations[j].y);

      // calculate particle weight
      double std_x = std_landmark[0];
      double std_y = std_landmark[1];
      double observation_weight =
        (1.0 / (2 * M_PI * std_x * std_y)) *
        exp(-(pow(prediction_x - observation_x, 2) / (2 * pow(std_x, 2)) +
             (pow(prediction_y - observation_y, 2) / (2 * pow(std_y, 2)))));
      particles[i].weight *= observation_weight;
      weights[i] = particles[i].weight;
    }
    
    // set associations
    SetAssociations(particles[i], associations, sense_x, sense_y);
  }
  return;
}

void ParticleFilter::resample()
{
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  std::vector<Particle> resampled_particles(num_particles);
  default_random_engine gen;

  // use discrete distribution to return particles by weight
  discrete_distribution<int> index(weights.begin(), weights.end());
  for(int i = 0; i < num_particles; i++)
  {
    resampled_particles[i] = particles[index(gen)];
  }

  // assign the resampled particles to the particle vector
  particles = resampled_particles;

  return;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
  const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
	
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
	return(particle);
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
