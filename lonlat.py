import numpy as np

def pixel_to_latlon_matrix(image_resolution, camera_params, drone_params):

    x_coords, y_coords = np.meshgrid(np.arange(image_resolution[0]), np.arange(image_resolution[1]))
    

    x_pixel = (x_coords - image_resolution[0] / 2) * camera_params['sensor_size'][0] / image_resolution[0]
    y_pixel = (y_coords - image_resolution[1] / 2) * camera_params['sensor_size'][1] / image_resolution[1]
    

    Z_c = camera_params['focal_length'] / 1000
    X_c = x_pixel * Z_c / camera_params['sensor_size'][0]
    Y_c = y_pixel * Z_c / camera_params['sensor_size'][1]

    pitch, roll, yaw = np.radians(drone_params['angles'])
    R = rotation_matrix(pitch, roll, yaw)
    world_coords = np.dot(R, np.array([X_c.flatten(), Y_c.flatten(), Z_c * np.ones_like(X_c.flatten())]))
    

    drone_lat = drone_params['latitude']
    drone_lon = drone_params['longitude']
    
    latlon_matrix = np.zeros((image_resolution[1], image_resolution[0], 2))
    

    dX = world_coords[0, :].reshape(image_resolution[1], image_resolution[0])
    dY = world_coords[1, :].reshape(image_resolution[1], image_resolution[0])
    
    lat, lon = world_to_latlon_batch(dX, dY, drone_lat, drone_lon)
    
    latlon_matrix[:, :, 0] = lat
    latlon_matrix[:, :, 1] = lon
    

    image_width_meters = camera_params['sensor_size'][0] * image_resolution[0] / camera_params['focal_length'] / 1000
    image_height_meters = camera_params['sensor_size'][1] * image_resolution[1] / camera_params['focal_length'] / 1000
    
    return latlon_matrix, image_width_meters, image_height_meters

def rotation_matrix(pitch, roll, yaw):
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    return np.dot(R_z, np.dot(R_y, R_x))

def world_to_latlon_batch(dX, dY, drone_lat, drone_lon):
    a = 6378137.0  # Semi-major axis in meters
    f = 1 / 298.257223563  # Flattening
    b = a * (1 - f)  # Semi-minor axis
    
    lat_rad = np.radians(drone_lat)
    
    N = a / np.sqrt(1 - (2*f - f**2) * np.sin(lat_rad)**2)
    
    m_per_deg_lat = (np.pi / 180) * (a * (1 - f)) / (1 - (2*f - f**2) * np.sin(lat_rad)**2)**1.5
    m_per_deg_lon = (np.pi / 180) * (N * np.cos(lat_rad))
    
    delta_lat = dY / m_per_deg_lat
    delta_lon = dX / m_per_deg_lon
    
    object_lat = drone_lat + delta_lat
    object_lon = drone_lon + delta_lon
    
    return object_lat, object_lon


from PIL import Image  
  


from PIL import Image  
from PIL import ExifTags  
  
def get_exif_data(image):  
    exif_data = {}  
    info = image._getexif()  
    if info:  
        for tag, value in info.items():  
            print(tag, value)
            
            decoded = ExifTags.TAG_NAME.get(tag, tag)  
            exif_data[decoded] = value  
    return exif_data  
  
# 打开图片  
image = Image.open(r' ')
  
info = image._getexif()  

ds1=[float(s)   for s in info[34853][2]]
ds2=[float(s)   for s in info[34853][4]]

lon=ds2[0]+ds2[1]/60+ds2[2]/3600
lat=ds1[0]+ds1[1]/60+ds1[2]/3600



# 示例使用
camera_params = {
    'focal_length': 7,
    'sensor_size': (36.0, 28.0),
    'image_resolution': (8064, 6048)
}

drone_params = {
    'latitude': lat,
    'longitude': lon,
    'altitude': info[34853][6],
    'angles': (0, 0, 0)
}

latlon_matrix, image_width_meters, image_height_meters = pixel_to_latlon_matrix(camera_params['image_resolution'], camera_params, drone_params)

print(f'LatLon Matrix Shape: {latlon_matrix.shape}')
print(f'Image Width: {image_width_meters:.2f} meters')
print(f'Image Height: {image_height_meters:.2f} meters')

