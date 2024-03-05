from isaacgym import gymapi, gymtorch, gymutil
from torchvision.utils import save_image
import numpy as np
import torch
from PIL import Image
import cv2

gym = gymapi.acquire_gym()

###### Parameter in simulation
# get default set of parameters
sim_params = gymapi.SimParams()

# set common parameters
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# set PhysX-specific parameters
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
# sim_params.use_gpu_pipeline = True

# create sim with these parameters
compute_device_id = 0
graphics_device_id = 0
physics_engine = gymapi.SimType.SIM_PHYSX
sim = gym.create_sim(compute_device_id, graphics_device_id, physics_engine, sim_params)

###### Ground Plane
# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!

# create the ground plane
gym.add_ground(sim, plane_params)

###### Asset
# load asset
asset_root = "../../resources/robots"
asset_file = "anymal_c/urdf/anymal_c.urdf"

# set asset option
asset_options = gymapi.AssetOptions()
# asset_options.fix_base_link = True
asset_options.flip_visual_attachments = True
asset_options.armature = 0.01

# load asset
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

print(gym.find_asset_rigid_body_index(asset, "depth_camera_front_camera")) # get index by name of rigid body in asset

###### Camera Sensor set properties
# define properties
camera_props = gymapi.CameraProperties()
camera_props.width = 512
camera_props.height = 340
camera_props.horizontal_fov = 87.3
camera_props.near_plane = 0.3
camera_props.far_plane = 3
camera_props.enable_tensors = True

###### Create env and actor handles
# cache useful handles
num_envs = 2
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 0.6)
euler = gymapi.Quat.from_euler_zyx(0, 0, 0)
pose.r = gymapi.Quat(euler.x, euler.y, euler.z, euler.w)
num_per_row = 3

envs = []
actor_handles = []
camera_handles = []
camera_tensors = []

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)
    
    # set actor properties
    props = gym.get_actor_dof_properties(env, actor_handle)
    props["driveMode"].fill(gymapi.DOF_MODE_NONE)
    props["stiffness"].fill(1000.0)
    props["damping"].fill(1000.0)
    gym.set_actor_dof_properties(env, actor_handle, props)
    
    #create camera sensor
    camera_handle = gym.create_camera_sensor(env, camera_props)
    camera_handles.append(camera_handle)
    
    rigid_handle = gym.get_actor_rigid_body_handle(env, actor_handle, 0)
    face_front_handle = gym.get_actor_rigid_body_handle(env, actor_handle, 43)
    # attach camera to the actor body
    beta = 0.04715
    gamma = 0.0292
    theta = 0.523598775598

    x = 0.4145 + beta
    z = -gamma
    local_transform = gymapi.Transform(p=gymapi.Vec3(x,0,z), r=gymapi.Quat.from_euler_zyx(0, 0.523598775598, 0))
    
    # local_transform1.p = gymapi.Vec3(0.4145, 0, 0)
    # local_transform1.r = gymapi.Quat.from_euler_zyx(0, 0, 0)
    # local_transform2 = gymapi.Transform()
    # local_transform2.p = gymapi.Vec3(0, 0, 0)
    # local_transform2.r = gymapi.Quat.from_euler_zyx(0, 0.523598775598, 0)
    # local_transform3 = gymapi.Transform()
    # local_transform3.p = gymapi.Vec3(0.04715, 0,-0.0292)
    # local_transform3.r = gymapi.Quat.from_euler_zyx(0, 0, 0)
    # local_transform = gymapi.Transform()
    # local_transform3
    # local_transform3.r = gymapi.Quat.from_euler_zyx(0, 0.523598775598, 0)
    gym.attach_camera_to_body(camera_handle, env, rigid_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
    
    # define camera tensor
    # camera_tensor = gym.get_camera_image_gpu_tensor(sim, env, camera_handle, gymapi.IMAGE_COLOR)
    camera_tensor = gym.get_camera_image_gpu_tensor(sim, env, camera_handles[0], gymapi.IMAGE_DEPTH)
    torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor) # the tensor will update auto
    print(torch_camera_tensor.shape)
    camera_tensors.append(torch_camera_tensor)

tt = torch.stack(camera_tensors)
print(tt.shape)
depth_camera_handle = gym.get_actor_rigid_body_handle(env, actor_handle, 44)
base_handle = gym.get_actor_rigid_body_handle(env, actor_handle, 0)
sphere_camera = gymutil.WireframeSphereGeometry(0.01, 10, 10, None, color=(1, 1, 0))
sphere_ref = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 0))
###### Visualize simulation
# create viwer and set properties
viewer_props = gymapi.CameraProperties()
viewer_props.horizontal_fov = 90.0
viewer_props.width = 1920
viewer_props.height = 1080
viewer = gym.create_viewer(sim, viewer_props)

# Point camera at environments
viwer_pos = gymapi.Vec3(4.0, 4.0, 2.0)
view_target = gymapi.Vec3(0.0, 2.0, 1.0)
gym.viewer_camera_look_at(viewer, None, viwer_pos, view_target)

img_path = "./img_save"
i = 0

# Draw frame for testing transformation
axes_geom = gymutil.AxesGeometry(scale=1, pose=None)


# loop for visualization
while not gym.query_viewer_has_closed(viewer):
    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Step rendering
    gym.step_graphics(sim)
    
    gym.render_all_camera_sensors(sim) # !!! dont for get to update data in tensor
    gym.start_access_image_tensors(sim)

    tt = torch.cat(camera_tensors)

    # move data to cpu
    # depth_image = tt.cpu().numpy()
    # -inf implies no depth value, set it to zero. output will be black.
    tt[tt == -np.inf] = 0
    # clamp depth image to 10 meters to make output image human friendly
    tt[tt < -1] = -1
    # flip the direction so near-objects are light and far objects are dark
    normalized_depth = -255.0*(tt/torch.min(tt + 1e-4))
    normalized_dept_numpy = normalized_depth.cpu().numpy()
    
    # for e in range(tt.shape[0]):
    
    # im = Image.fromarray(tt[:, :, :3].cpu().numpy())
    print(normalized_dept_numpy)
    cv2.imshow('Frame', normalized_dept_numpy.astype(np.uint8))
    cv2.waitKey(1) 

    im = Image.fromarray(normalized_dept_numpy.astype(np.uint8), mode="L")
    im.save("{}/frame{}.png".format(img_path, i))
    i += 1
    
    cam_position = gym.get_camera_transform(sim, env, camera_handle)
    rigid_ref_position = gym.get_rigid_transform(env, depth_camera_handle)
    rigid_face_front = gym.get_rigid_transform(env, face_front_handle)
    rigid_base = gym.get_rigid_transform(env, base_handle)
    
    gym.clear_lines(viewer)
    
    point = cam_position.p
    # transbase_ore = gymapi.Transform( r=cam_position.r)
    transface = gymapi.Transform(r=cam_position.r, p=gymapi.Vec3(0.4145, 0, 0))
    point = transface.transform_point(point)
    # point = transbase_ore.transform_vector(point)
    point = transface.transform_point(point)
    
    # print(cam_position.p)
    print("base p", rigid_base.p)
    print("face p", rigid_face_front.p)
    print("cam p", cam_position.p)
    print("r cam p", rigid_ref_position.p)
    
    print("base r", rigid_base.r)
    print("face r", rigid_face_front.r)
    print("cam p", cam_position.r)
    print("r cam r", rigid_ref_position.r)
   


    # gymutil.draw_lines(axes_geom, gym, viewer, env, axes_pose)
    gymutil.draw_lines(sphere_camera, gym, viewer, env, cam_position) 
    gymutil.draw_lines(sphere_ref, gym, viewer, env, rigid_ref_position) 
    
    gym.end_access_image_tensors(sim)
    
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
cv2.destroyAllWindows() 