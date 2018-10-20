import numpy as np
import cv2

scale = 10

# Identify pixels above the threshold
def color_thresh(img, rgb_lower_thresh=(160, 160, 160), rgb_upper_thresh=(255,255,255)):
    color_select = cv2.inRange(img, rgb_lower_thresh, rgb_upper_thresh)
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    ypos, xpos = binary_img.nonzero()
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel

# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    return xpix_rotated, ypix_rotated

# Define a function to translate pixels
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    return warped

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # 1) Define source and destination points for perspective transform 
    # TODO: remove from the loop
    dst_size = 5
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                      [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                      [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                      [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                      ])

    # 2) Apply perspective transform
    warped = perspect_transform(Rover.img, source, destination)
    fov_mask = perspect_transform(np.ones_like(Rover.img[:,:,0]), source, destination)
    # TODO: Improve FOV    
    fov_mask[:10,:] = 0
    fov_mask[:,:20] = 0
    fov_mask[:,-20:] = 0

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    navigable = color_thresh(warped, rgb_lower_thresh=(160,160,160), rgb_upper_thresh=(255,255,255)) * fov_mask
    obstacles = color_thresh(warped, rgb_lower_thresh=(0,0,0), rgb_upper_thresh=(160,160,160)) * fov_mask
    rock_sample = color_thresh(warped, rgb_lower_thresh=(100,100,0), rgb_upper_thresh=(255,255,0)) * fov_mask

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:,:,0] = obstacles
    Rover.vision_image[:,:,1] = rock_sample
    Rover.vision_image[:,:,2] = navigable

    # 5) Convert map image pixel values to rover-centric coords
    obs_xpix, obs_ypix = rover_coords(obstacles)
    nav_xpix, nav_ypix = rover_coords(navigable)
    roc_xpix, roc_ypix = rover_coords(rock_sample)

    # 6) Convert rover-centric pixel values to world coordinates
    obstacle_x_world, obstacle_y_world = pix_to_world(obs_xpix, obs_ypix, \
            Rover.pos[0], Rover.pos[1], \
            Rover.yaw, Rover.worldmap.shape[0], scale)
    navigable_x_world, navigable_y_world = pix_to_world(nav_xpix, nav_ypix, \
            Rover.pos[0], Rover.pos[1], \
            Rover.yaw, Rover.worldmap.shape[0], scale)
    rock_x_world, rock_y_world = pix_to_world(roc_xpix, roc_ypix, \
            Rover.pos[0], Rover.pos[1], \
            Rover.yaw, Rover.worldmap.shape[0], scale)

    # 7) Update Rover worldmap (to be displayed on right side of screen)
    print('yaw   ' + str(Rover.yaw))
    print('pitch ' + str(Rover.pitch))
    print('roll  ' + str(Rover.roll))
    print('------------')
    if (Rover.pitch > 359 or Rover.pitch < 1) and (Rover.roll > 358.8 or Rover.roll < 1.2):
        Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    else:
        print("Discard reading due to instability")

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    rover_centric_pixel_distances, rover_centric_angles = to_polar_coords(nav_xpix, nav_ypix)
    Rover.nav_dists = rover_centric_pixel_distances
    Rover.nav_angles = rover_centric_angles
    #Rover.samples_pos = (np.mean(rock_x_world), np.mean(rock_y_world))
    rover_centric_sample_distances, rover_centric_sample_angles = to_polar_coords(roc_xpix, roc_ypix)
    Rover.sample_dists = rover_centric_sample_distances
    Rover.sample_angles = rover_centric_sample_angles


    return Rover
