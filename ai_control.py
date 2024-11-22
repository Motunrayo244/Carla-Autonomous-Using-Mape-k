#!/usr/bin/env python

import glob
import os
import sys
from collections import deque
import math
import numpy as np

try:
    sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import ai_knowledge as data
from ai_knowledge import Status
from agents.navigation.global_route_planner import GlobalRoutePlanner


# Executor is responsible for moving the vehicle around
# In this implementation it only needs to match the steering and speed so that we arrive at provided waypoints
# BONUS TODO: implement different speed limits so that planner would also provide speed target speed in addition to direction
class Executor(object):
  def __init__(self, knowledge, vehicle):
    self.vehicle = vehicle
    self.knowledge = knowledge
    self.target_pos = knowledge.get_location()
    self.curr_wp =0
    
  #Update the executor at some intervals to steer the car in desired direction
  def update(self, time_elapsed):
    status = self.knowledge.get_status()
    #TODO: this needs to be able to handle
    if status == Status.DRIVING:
      dest = self.knowledge.get_current_destination()
      target_speed = self.knowledge.retrieve_data('target_speed')
      route_rotations = self.knowledge.retrieve_data('route_rotations')
      brake = self.knowledge.retrieve_data('brake')
      self.update_control(dest,
                          {'target_speed': target_speed,
                            'rotations': route_rotations,
                            'brake': brake},
                          time_elapsed)
    elif status == Status.HEALING:
      collision_risk = self.knowledge.retrieve_data('collision_risk')
      self.update_control(None,
                          {'target_speed': 0,
                           'collision_risk': collision_risk}, 
                          time_elapsed)
  # TODO: steer in the direction of destination and throttle or brake depending on how close we are to destination
  # TODO: Take into account that exiting the crash site could also be done in reverse,
  # so there might need to be additional data passed between planner and executor, 
  # or there needs to be some way to tell this that it is ok to drive in reverse
  # during HEALING and CRASHED states.
  # An example is additional_vars, that could be a list with parameters that can tell us 
  # which things we can do (for example going in reverse)
  
  
  def update_control(self, destination, additional_vars, delta_time):
    PREFERRED_SPEED = additional_vars.get('target_speed', 0)/36 # convert km/h to m/s
    route_rotations = additional_vars.get('rotations', [])
    collision_risk = additional_vars.get('collision_risk', False)
    route = self.knowledge.retrieve_data('route')
    brake = additional_vars.get('brake', 0.0)

    if collision_risk:
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.steer = 0.5  # Turn sharply to avoid collision
        control.brake = 1.0
        control.hand_brake = False
        self.vehicle.apply_control(control)
        return

    if not route or len(route) == 0:
      self.knowledge.update_status(Status.ARRIVED)
      control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
      self.vehicle.apply_control(control)       
      return
      
# # Check if vehicle is close to the destination
    if destination and self.knowledge.arrived_at(destination):        
        control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        self.vehicle.apply_control(control)
        
        self.knowledge.update_status(Status.ARRIVED)
        return
    
    if self.vehicle.get_location().distance(destination) <5:
      control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
      self.vehicle.apply_control(control)
      self.knowledge.update_status(Status.ARRIVED)
      return
      
    # Update current waypoint index if close enough to the current target waypoint
    while self.curr_wp < len(route) and self.vehicle.get_transform().location.distance(route[self.curr_wp]) < 4 :
        self.curr_wp += 1

    if self.curr_wp >= len(route):      
      self.knowledge.update_status(Status.ARRIVED) 
      return
   

    # Get the vehicle's current location and the next waypoint
    current_location = self.vehicle.get_transform().location
    next_waypoint_location = route[self.curr_wp]
    next_waypoint_rotation = route_rotations[self.curr_wp]
    
    # Calculate the direction vector to the next waypoint
    dest_vector = np.array([next_waypoint_location.x - current_location.x,
                            next_waypoint_location.y - current_location.y, 0])
    dest_distance = np.linalg.norm(dest_vector)
    dest_vector_normalized = dest_vector / dest_distance if dest_distance != 0 else np.array([0, 0, 0])

    # Get the vehicle's current forward vector
    current_transform = self.vehicle.get_transform()
    current_forward_vector = current_transform.get_forward_vector()
    current_forward = np.array([current_forward_vector.x, current_forward_vector.y, 0])
    current_forward_normalized = current_forward / np.linalg.norm(current_forward)

    # Calculate the angle between the vehicle's forward vector and the destination vector
    dot_product = np.dot(current_forward_normalized, dest_vector_normalized)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle = np.arccos(dot_product)
    cross_product = np.cross(current_forward_normalized, dest_vector_normalized)
    steer_direction = np.sign(cross_product[2])

    # Calculate dynamic braking threshold based on speed
    current_velocity = self.vehicle.get_velocity()
    current_speed = np.linalg.norm(np.array([current_velocity.x, current_velocity.y, 0]))
    max_deceleration = 8  # assumed comfortable deceleration in m/s^2
    basic_threshold = 0.4 # meters
    braking_threshold = basic_threshold + (current_speed ** 2) / (2 * max_deceleration)

    control = carla.VehicleControl()
    braking_intensity = max(0, (braking_threshold - dest_distance) / braking_threshold)

  
    
    if dest_distance < braking_threshold:
        print("Braking!")
        control.brake = np.clip(braking_intensity, 0.5, 1)
        control.throttle = 0
    else:
        control.throttle = min(PREFERRED_SPEED / current_speed, 1.0) if current_speed > 0 else 0.5
        control.brake = brake
    
    # Smooth steering interpolation
    control.steer = steer_direction * angle / np.pi
    control.hand_brake = False

    # Adjust vehicle orientation to match the destination rotation
    if additional_vars and 'rotations' in additional_vars:
        current_yaw = current_transform.rotation.yaw
        destination_yaw = next_waypoint_rotation.yaw
        yaw_diff = destination_yaw - current_yaw

        # Normalize the yaw difference
        if yaw_diff > 180:
            yaw_diff -= 360
        elif yaw_diff < -180:
            yaw_diff += 360

        control.steer += yaw_diff / 180  # Adjust steer based on yaw difference

    self.vehicle.apply_control(control)
    
  
  
# Planner is responsible for creating a plan for moving around
# In our case it creates a list of waypoints to follow so that vehicle arrives at destination
# Alternatively this can also provide a list of waypoints to try avoid crashing or 'uncrash' itself
class Planner(object):
  def __init__(self, knowledge,world):
    self.knowledge = knowledge
    self.world = world
    self.path = deque([])

  # Create a map of waypoints to follow to the destination and save it
  def make_plan(self, source, destination):
    self.path = self.build_path(source,destination) # build the route using waypoint
    self.update_plan()
    self.knowledge.update_data('final_destination', destination)
    self.knowledge.update_destination(self.get_current_destination())
  
  # Function that is called at time intervals to update ai-state
  def update(self, time_elapsed):
    self.update_plan()
    self.knowledge.update_destination(self.get_current_destination())
  
  #Update internal state to make sure that there are waypoints to follow and that we have not arrived yet
  def update_plan(self):
    # if len(self.path) == 0:
    #   return 
    route = self.knowledge.retrieve_data('route')
    route_rotations = self.knowledge.retrieve_data('route_rotations')

    if not route:
        return

    if self.knowledge.arrived_at(route[0]):
        route.popleft()
        route_rotations.popleft()
        self.knowledge.update_data('route', route)
        self.knowledge.update_data('route_rotations', route_rotations)

    if len(route) == 0:
      self.knowledge.update_data ('target_speed',0.0)
      self.knowledge.update_data ('brake', 1.0)
      self.knowledge.update_status(Status.ARRIVED)
    else:
        self.knowledge.update_status(Status.DRIVING)
        self.draw_route(route)

  #get current destination 
  def get_current_destination(self):
    status = self.knowledge.get_status()
    #if we are driving, then the current destination is next waypoint
    if status == Status.DRIVING:
      #TODO: Take into account traffic lights and other cars
      return self.path[0]
    if status == Status.ARRIVED:
      self.knowledge.update_data ('target_speed',0.0)
      self.knowledge.update_data ('brake', 1.0)
      return self.knowledge.get_location()
    if status == Status.HEALING:
      #Implement crash handling. Probably needs to be done by following waypoint list to exit the crash site.
      #Afterwards needs to remake the path.
      
      return self.knowledge.get_location()
    if status == Status.CRASHED:
      #TODO: implement function for crash handling, should provide map of wayoints to move towards to for exiting crash state. 
      #You should use separate waypoint list for that, to not mess with the original path. 
      print("Handling crash, stopping vehicle.")
      self.stop_vehicle()
      self.exit_crash_site()
      self.knowledge.update_status(Status.HEALING)
      return self.knowledge.get_location()
    #otherwise destination is same as current position
    return self.knowledge.get_location()
  
 
  
  #TODO: Implementation
  def build_path(self, source, destination):
    self.path = deque([])
    
    source_location = self.world.get_map().get_waypoint(source)
    destination_location = self.world.get_map().get_waypoint(destination)

    # Create path of waypoints from source to destination
    sampling_resolution = 0.5
    grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution)
    way_points = grp.trace_route(source_location.transform.location, destination_location.transform.location)
    route = deque([])
    rotations = deque([])
        
    for wp in way_points:
      current_waypoint = wp[0]
      if current_waypoint.lane_change in (carla.LaneChange.Right, carla.LaneChange.Both):
        right_waypoint = current_waypoint.get_right_lane()
        if right_waypoint and right_waypoint.lane_type == carla.LaneType.Driving:
            current_waypoint = right_waypoint

      elif current_waypoint.lane_change in (carla.LaneChange.Left, carla.LaneChange.Both):
        left_waypoint = current_waypoint.get_left_lane()
        if left_waypoint and left_waypoint.lane_type == carla.LaneType.Driving:
            current_waypoint = left_waypoint
            
      route.append(wp[0].transform.location)
      rotations.append(wp[0].transform.rotation)
           

    # Update knowledge with waypoints and rotations
    self.knowledge.update_data('route', route)
    self.knowledge.update_data('route_rotations', rotations)

    return route

  
  def draw_route(self, route, seconds=5.0):
    draw_color = carla.Color(r=0, g=255, b=0) if len(route) < 40 else carla.Color(r=255, g=0, b=0)
    for i in range(0, len(route), 3):  # Increment by 3 to avoid IndexError
        if i < len(route):
            self.world.debug.draw_string(route[i], '^', draw_shadow=False,
                                          color=draw_color, life_time=seconds, persistent_lines=True)
    return None

  def exit_crash_site(self):
    crash_location = self.vehicle.get_location()
    # Find a nearby waypoint that is not in the direction of the crash
    exit_waypoint = self.find_exit_waypoint(crash_location)
    if exit_waypoint:
        self.knowledge.update_destination(exit_waypoint.location)
        self.curr_wp = 0  # Reset waypoint index for new path
        print("Exiting crash site towards: ", exit_waypoint.location)
        
  def find_exit_waypoint(self, crash_location):
    map = self.world.get_map()
    # Get a waypoint behind the vehicle as an example
    vehicle_transform = self.vehicle.get_transform()
    backward_vector = vehicle_transform.rotation.get_forward_vector() * -1
    backward_location = carla.Location(x=crash_location.x + backward_vector.x * 10,
                                       y=crash_location.y + backward_vector.y * 10,
                                       z=crash_location.z)
    return map.get_waypoint(backward_location)

def remake_path(self):
    current_location = self.vehicle.get_location()
    destination = self.knowledge.retrieve_data('final_destination')  
    self.make_plan(current_location, destination)
    
    print("Path recalculated from crash site to destination.")




