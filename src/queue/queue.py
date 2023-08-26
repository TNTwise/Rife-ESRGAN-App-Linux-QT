from PyQt5.QtWidgets import QFileDialog
from threading import *
from src.settings import *
from src.return_data import *
ManageFiles.create_folder(f'{thisdir}/files/')
import src.workers as workers
#import src.get_models as get_models
from src.messages import *
def addToQueue(self):
    self.queueFile = QFileDialog.getOpenFileName(self, 'Open File', f'{homedir}',"Video files (*.mp4);;All files (*.*)")[0]
    if self.queueFile != '':
        self.QueueList.append(self.queueFile)
        self.queueVideoName = VideoName.return_video_name(self.queueFile)
        self.ui.QueueListWidget.addItem(self.queueVideoName)
        self.ui.QueueListWidget.show()