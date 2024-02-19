from isaacgym import gymapi, gymtorch
from torchvision.utils import save_image
import numpy as np
import torch
from PIL import Image

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

###### Camera Sensor set properties
# define properties
camera_props = gymapi.CameraProperties()
camera_props.width = 512
camera_props.height = 512
camera_props.enable_tensors = True

###### Create env and actor handles
# cache useful handles
num_envs = 1
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
    props["damping"].fill(100.0)
    gym.set_actor_dof_properties(env, actor_handle, props)
    
    #create camera sensor
    camera_handle = gym.create_camera_sensor(env, camera_props)
    camera_handles.append(camera_handle)
    # attach camera to the actor body
    local_transform = gymapi.Transform()
    local_transform.p = gymapi.Vec3(0,0,0)
    local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0.3,0,0), np.radians(0))
    gym.attach_camera_to_body(camera_handle, env, actor_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

# define 
camera_tensor = gym.get_camera_image_gpu_tensor(sim, env, camera_handles[0], gymapi.IMAGE_DEPTH)
torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor) # the tensor will update auto

###### Visualize simulation
# create viwer and set properties
viewer_props = gymapi.CameraProperties()
viewer_props.horizontal_fov = 75.0
viewer_props.width = 1920
viewer_props.height = 1080
viewer = gym.create_viewer(sim, viewer_props)

# Point camera at environments
cam_pos = gymapi.Vec3(-4.0, 4.0, 2.0)
cam_target = gymapi.Vec3(0.0, 2.0, 1.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

img_path = "./img_save"
i = 0

# loop for visualization
while not gym.query_viewer_has_closed(viewer):
    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Step rendering
    gym.step_graphics(sim)
    
    gym.render_all_camera_sensors(sim) # !!! dont for get to update data in tensor
    gym.start_access_image_tensors(sim)
    # print(torch.max(torch_camera_tensor))
    # print(torch_camera_tensor.dtype)
    # print(torch_camera_tensor[:, :, :3])
    depth_image = torch_camera_tensor.cpu().numpy()
    
    # -inf implies no depth value, set it to zero. output will be black.
    depth_image[depth_image == -np.inf] = 0
    # clamp depth image to 10 meters to make output image human friendly
    depth_image[depth_image < -10] = -10
    # flip the direction so near-objects are light and far objects are dark
    normalized_depth = -255.0*(depth_image/np.min(depth_image + 1e-4))
    
    # print(torch_camera_tensor.cpu().numpy())

    im = Image.fromarray(normalized_depth.astype(np.uint8), mode="L")
    im.save("{}/{}.png".format(img_path, i))
    i += 1
    
    gym.end_access_image_tensors(sim)
    
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
