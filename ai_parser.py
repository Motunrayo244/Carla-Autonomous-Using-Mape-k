#!/usr/bin/env python

import glob
import os
import sys

try:
    sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import weakref
import carla
import ai_knowledge as data
from ai_knowledge import Status
import numpy as np
import cv2
import math

IM_WIDTH = 640
IM_HEIGHT = 360
# Monitor is responsible for reading the data from the sensors and telling it to the knowledge
# TODO: Implement other sensors (lidar and depth sensors mainly)
# TODO: Use carla API to read whether car is at traffic lights and their status, update it into knowledge
class Monitor(object):
  def __init__(self, knowledge,vehicle):
    self.vehicle = vehicle
    self.knowledge = knowledge
       
    
    self.knowledge.update_data('location', self.vehicle.get_transform().location)
    self.knowledge.update_data('rotation', self.vehicle.get_transform().rotation)
    self.knowledge.update_data('vehicle_id', self.vehicle.id)
    self.knowledge.update_data('world', self.vehicle.get_world())
    self.knowledge.update_data('velocity', self.vehicle.get_velocity())
    
    self.world = self.vehicle.get_world()
    self.sensors = []
    self.setup_sensors(self.world)
    
    
  #Function that is called at time intervals to update ai-state
  def update(self, time_elapsed):
    # Update the position of vehicle into knowledge
    self.knowledge.update_data('location', self.vehicle.get_transform().location)
    self.knowledge.update_data('rotation', self.vehicle.get_transform().rotation)
    self.knowledge.update_data('vehicle_id', self.vehicle.id)
    self.knowledge.update_data('velocity', self.vehicle.get_velocity())
    self.check_traffic_lights()
    

  def setup_sensors(self, world):
    weak_self = weakref.ref(self)
    lane_bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
    self.lane_detector = world.spawn_actor(lane_bp, carla.Transform(), attach_to=self.vehicle)
    self.lane_detector.listen(lambda event: self._on_invasion(weak_self, event))
    self.sensors.append(self.lane_detector)
    
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('range', '100')  # Range in meters
    lidar_bp.set_attribute('points_per_second', '100000')
    self.lidar = world.spawn_actor(lidar_bp,
                                   carla.Transform(carla.Location(x=0, z=2.5)), 
                                   attach_to=self.vehicle)
    self.lidar.listen(lambda event: self._on_lidar(weak_self, event))
    self.sensors.append(self.lidar)
    
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    # change the dimensions of the image
    camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
    camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
    camera_bp.set_attribute('fov', '140')
    self.camera = world.spawn_actor(camera_bp, 
                                    carla.Transform(carla.Location(x=-5, z=2.5),
                                    carla.Rotation(pitch=-15.0, yaw=0.0, roll=0.0)), 
                                    attach_to=self.vehicle)
    self.camera.listen(lambda image: self._on_camera(image))
    self.sensors.append(self.camera)
    
    # #Collision sensor
    # collision_bp = world.get_blueprint_library().find('sensor.other.collision')
    # collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
    # collision_sensor.listen(lambda event: self._on_collision(weak_self,event))
    
    
    
  @staticmethod
  def _on_camera(image):
      i = np.array(image.raw_data)  # convert to an array
      i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))  # was flattened, so we're going to shape it.
      i3 = i2[:, :, :3]  # remove the alpha (basically, remove the 4th index  of every pixel. Converting RGBA to RGB)
      cv2.imshow("", i3)  # show it.
      cv2.waitKey(1)
      return i3/255.0  # normalize  
  
  
  @staticmethod
  def _on_invasion(weak_self, event):
    self = weak_self()
    if not self:
      return
    self.knowledge.update_data('lane_invasion',event.crossed_lane_markings)
    
  @staticmethod
  def _on_lidar(weak_self, event):
    self = weak_self()
    if not self:
      return
    data = np.frombuffer(event.raw_data, dtype=np.dtype('f4'))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))  
    self.knowledge.update_data('lidar_data', data)
    
  @staticmethod
  def _on_collision(weak_self,event):
    self = weak_self()
    if not self:
      return
    print("Collision detected: ", event.other_actor.type_id)
    self.knowledge.update_status(Status.CRASHED)
    
  def check_traffic_lights(self):
        relevant_traffic_lights = self.get_relevant_traffic_lights()
        self.knowledge.update_data('relevant_traffic_lights', relevant_traffic_lights)

  def get_relevant_traffic_lights(self):
        vehicle_location = self.vehicle.get_transform().location
        vehicle_waypoint = self.vehicle.get_world().get_map().get_waypoint(vehicle_location)
        traffic_lights = self.vehicle.get_world().get_actors().filter('traffic.traffic_light')
        relevant_traffic_lights = []

        for traffic_light in traffic_lights:
            tl_location = traffic_light.get_transform().location
            tl_waypoint = self.vehicle.get_world().get_map().get_waypoint(tl_location)

            if vehicle_waypoint.is_junction and tl_waypoint.is_junction:
                if self.is_traffic_light_in_path(vehicle_waypoint, tl_waypoint):
                    relevant_traffic_lights.append(traffic_light)

        return relevant_traffic_lights

  def is_traffic_light_in_path(self, vehicle_waypoint, tl_waypoint):
      next_waypoints = vehicle_waypoint.next(20)
      for wp in next_waypoints:
          if wp.transform.location.distance(tl_waypoint.transform.location) < 10:
              return True
      return False
    
    
  
  def destroy_sensors(self):
    for sensor in self.sensors:
        if sensor.is_alive:
          print(f'Destroying sensor{sensor}')
          sensor.stop()  # Stop the sensor from listening
          sensor.destroy()  # Remove the sensor from the simulation
    self.sensors = []

# Analyser is responsible for parsing all the data that the knowledge has received from Monitor and turning it into something usable
# TODO: During the update step parse the data inside knowledge into information that could be used by planner to plan the route
class Analyser(object):
  def __init__(self, knowledge):
    self.knowledge = knowledge

  #Function that is called at time intervals to update ai-state
  def update(self, time_elapsed):
        lidar_data = self.knowledge.retrieve_data('lidar_data')
        relevant_traffic_lights = self.knowledge.retrieve_data('relevant_traffic_lights')
        vehicle_location = self.knowledge.retrieve_data('location')
        target_speed = 54
        brake = 0.0

        if lidar_data is not None:
            if self.process_lidar_data(lidar_data):
                self.knowledge.update_status(Status.HEALING)

        if relevant_traffic_lights:
            for traffic_light in relevant_traffic_lights:
                print(f'Traffic light state: {traffic_light.get_state()}')
                tl_location = traffic_light.get_transform().location
                distance_to_tl = vehicle_location.distance(tl_location)
                if traffic_light.get_state() == carla.TrafficLightState.Red and distance_to_tl < 30:
                    target_speed = 0
                    brake = 1.0
                    break

        self.knowledge.update_data('target_speed', target_speed)
        self.knowledge.update_data('brake', brake)
        return
      
      
  def process_lidar_data(self, lidar_data):
    vehicle_location = self.knowledge.retrieve_data('location')
    vehicle_velocity = self.knowledge.retrieve_data('velocity')  # velocity is stored in knowledge

    if vehicle_velocity is None:
        vehicle_velocity = carla.Vector3D(0, 0, 0)  # Default to stationary if no velocity data
    
    for point in lidar_data:
      x, y, z = float(point[0]), float(point[1]), float(point[2])
      point_location = carla.Location(x=x, y=y, z=z)
      point_world = point_location + vehicle_location

      for actor in self.knowledge.retrieve_data('world').get_actors():
        if actor.id != self.knowledge.retrieve_data('vehicle_id') and 'vehicle' in actor.type_id:
          distance = actor.get_location().distance(point_world)
          if distance < 1.0:
            relative_velocity = actor.get_velocity() - vehicle_velocity
            if self.is_collision_likely(relative_velocity, distance):
              print(f'Potential collision detected with {actor.type_id}.\n Taking measures to avoid collision')
              self.knowledge.update_data('collision_risk', True)
              return True
    self.knowledge.update_data('collision_risk', False)
    return False
        
  @staticmethod
  def is_collision_likely(relative_velocity, distance):
    speed = np.linalg.norm(np.array([relative_velocity.x, relative_velocity.y, relative_velocity.z]))
    time_to_collision = distance / speed if speed != 0 else float('inf')
    return time_to_collision < 2.0  # Assuming a collision is likely if it could occur in less than 2 seconds