import time
import os
from time import sleep
from src.settings import Settings
def calculateETA(self):
        completed_iterations = int(self.files_processed)
                    
                    # Increment the completed iterations counter
                    
                    
        # Estimate the remaining time
        elapsed_time = time.time() - self.start_time
        time_per_iteration = elapsed_time / completed_iterations
        remaining_iterations = self.filecount - completed_iterations
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
        return f'ETA: {hours}:{minutes}:{seconds}'