import cv2
import glob
from tqdm import tqdm

def m_speed_change(path_in, path_out, scale_factor, color_flag):
    movie = cv2.VideoCapture(path_in)
 
    fps = int(movie.get(cv2.CAP_PROP_FPS))                                  
    fps_new = 120
    w = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH))                            
    h = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))                           
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')                    
    video = cv2.VideoWriter(path_out, fourcc, fps_new, (w, h), color_flag)  
 
    while True:
        ret, frame = movie.read()        
        video.write(frame)               

        if not ret:
            break
    movie.release()
    return
 
path_in = 'rgb-camera_removed_new/CIMG0256_004.MOV'         
path_out = 'video_out_256.mp4'     
scale_factor = 10           
color_flag = True              


num_list = ['05', '14', '18']


for i in tqdm(num_list):
    paths = glob.glob(f'rgb-camera_removed_new/*_0{i}.MOV')
    for j, path in enumerate(paths):
        print(path)
        path_in = path
        path_out = f'short_video_{j}_{i}.mp4'
        m_speed_change(path_in, path_out, scale_factor, color_flag)


        
