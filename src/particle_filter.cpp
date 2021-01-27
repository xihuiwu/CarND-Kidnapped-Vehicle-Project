/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

static std::default_random_engine generator;

#define EPS 0.00001

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  this->num_particles = 10;  // TODO: Set the number of particles
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  
  for (int i = 0; i < this->num_particles; i++){
    Particle p;
    p.id = i;
    p.x = dist_x(generator);
    p.y = dist_y(generator);
    p.theta = dist_theta(generator);
    p.weight = 1.0;
    this->particles.push_back(p);
  }
  this->is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::normal_distribution<double> dist_x(0.0, std_pos[0]);
  std::normal_distribution<double> dist_y(0.0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0.0, std_pos[2]);
  
  for (auto &p : this->particles){
    if (std::fabs(yaw_rate) >= EPS){
      p.x = p.x + (velocity/yaw_rate)*(sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
      p.y = p.y + (velocity/yaw_rate)*(cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
      p.theta = p.theta + yaw_rate*delta_t;
    }
    else{
      p.x = p.x + velocity*delta_t*cos(p.theta);
      p.y = p.y + velocity*delta_t*sin(p.theta);
    }

    p.x = p.x + dist_x(generator);
    p.y = p.y + dist_y(generator);
    p.theta = p.theta + dist_theta(generator);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> sensed, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the observed landmark that is cloest to each 
   *   particle sensed landmark and assign the id.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (auto &o : observations){
    double min_val = std::numeric_limits<double>::max();
    double d;
    for (auto &s : sensed){
      d = dist(o.x, o.y, s.x, s.y);
      if (d < min_val){
        min_val = d;
      	o.id = s.id;
      }
    }
  }
  
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  std::vector<double> new_weights;
  
  for (auto &p : this->particles){
    p.weight = 1.0;
    // find the landmarks that can be sensed by the particle 
    std::vector<LandmarkObs> sensed;
    for (auto &l : map_landmarks.landmark_list){
      double dx = l.x_f - p.x;
      double dy = l.y_f - p.y;
      /*
      if (dx*dx + dy*dy <= sensor_range*sensor_range){
        sensed.push_back(LandmarkObs{l.id_i, l.x_f, l.y_f});
      }
      */
      if (std::fabs(dx) <= sensor_range && std::fabs(dy) <= sensor_range){
      	sensed.push_back(LandmarkObs{l.id_i, l.x_f, l.y_f});
      }
    }
    
    // transfer observation from vehicle coordination to map coordination
    // i.e. get the coordination of the observed landmark in map coordination system
    std::vector<LandmarkObs> transformed_obs;
    for (auto &o : observations){
      double x = o.x*cos(p.theta) - o.y*sin(p.theta) + p.x;
      double y = o.x*sin(p.theta) + o.y*cos(p.theta) + p.y;
      transformed_obs.push_back(LandmarkObs{o.id, x, y});
    }
    
    // associate sensed landmarks and observations
    dataAssociation(sensed, transformed_obs);
    
    // calculate the particle's weight
    for (auto &o : transformed_obs){
      
      // find the associated landmark
      LandmarkObs associate;
      for (auto &s : sensed){
        if (o.id == s.id){
          associate = s;
          break;
        }
      }
      
      // calculate the probability density
      double x_val = o.x;
      double y_val = o.y;
      double mu_x = associate.x;
      double mu_y = associate.y;
      double sigma_x = std_landmark[0];
      double sigma_y = std_landmark[1];
      double w_obs = exp( - pow(x_val-mu_x,2)/(2*pow(sigma_x, 2)) - pow(y_val-mu_y,2)/(2*pow(sigma_y, 2)) ) / (2*M_PI*sigma_x*sigma_y);
      
      // multiply individual weight
      p.weight *= w_obs;
    }
    new_weights.push_back(p.weight);
  }
  this->weights = new_weights;
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::vector<Particle> res;
  std::vector<double> new_weights;
  std::discrete_distribution<int> idx_dist(this->weights.begin(), this->weights.end());
  
  
  for(int i = 0; i < this->num_particles; i++)
  {
    int rand_idx = idx_dist(generator);
    res.push_back(particles[rand_idx]);
    new_weights.push_back(particles[rand_idx].weight);
  }
  this->particles = res;
  this->weights = new_weights;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
