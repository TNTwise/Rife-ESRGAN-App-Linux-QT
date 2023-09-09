import time
import os
from time import sleep
def calculateETA(self,total_iterations):
        self.ETA=None
        
        start_time = time.time()
        while os.path.exists(f'{self.render_folder}/{self.videoName}_temp/'):
            if os.path.exists(f'{self.render_folder}/{self.videoName}_temp/input_frames/'):
                
                
                
                
                    # Do some work for each iteration
                    
                try:
                   
                        
                    
                    completed_iterations = int(self.files_processed)
                    
                    # Increment the completed iterations counter
                    sleep(1)
                    
                    # Estimate the remaining time
                    elapsed_time = time.time() - start_time
                    time_per_iteration = elapsed_time / completed_iterations
                    remaining_iterations = total_iterations - completed_iterations
                    remaining_time = remaining_iterations * time_per_iteration
                    remaining_time = int(remaining_time) 
                    # Print the estimated time remaining
                    #convert to hours, minutes, and seconds
                    hours = remaining_time // 3600
                    remaining_time-= 3600*hours
                    minutes = remaining_time // 60
                    remaining_time -= minutes * 60
                    seconds = remaining_time
                    if minutes < 10:
                        minutes = str(f'0{minutes}')
                    if seconds < 10:
                        seconds = str(f'0{seconds}')
                    self.ETA = f'ETA: {hours}:{minutes}:{seconds}'
                    
                except:
                    self.ETA = None
        return