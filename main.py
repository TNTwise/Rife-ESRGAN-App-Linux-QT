#!/usr/bin/python3
import src.getModels.select_models as sel_mod
from PyQt5 import QtWidgets, uic
import sys
from PyQt5.QtCore import QObject, QThread, pyqtSignal
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QMessageBox
from PyQt5.QtGui import QTextCursor, QPixmap,QIcon, QIntValidator
import PyQt5.QtCore as QtCore
import mainwindow
import os
from threading import *
from src.settings import *
from src.return_data import *
ManageFiles.create_folder(f'{thisdir}/files/')
import src.runAI.start as start
import src.workers as workers
import time
#import src.get_models as get_models
from time import sleep
import src.getModels.get_models as get_models
from multiprocessing import cpu_count
from src.messages import *

import pypresence
import src.onProgramStart
thisdir = os.getcwd()
homedir = os.path.expanduser(r"~")


class MainWindow(QtWidgets.QMainWindow):
         
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setMinimumSize(700, 550)
        self.ui = mainwindow.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowIcon(QIcon(f'{thisdir}/icons/logo v1.png'))
        self.ui.SettingsMenus.clicked.connect(self.settings_menu)
        self.gpuMemory=HardwareInfo.get_video_memory_linux()
        if self.gpuMemory == None:
            cannot_detect_vram(self)
        else:
            print(self.gpuMemory)
        src.onProgramStart.onApplicationStart(self)
        
        
        
        self.show()
    
    def calculateETA(self):
        self.ETA=None
        total_iterations = len(os.listdir(f'{self.render_folder}/{self.videoName}_temp/input_frames/')) * self.times
        for i in range(total_iterations):
            if os.path.exists(f'{self.render_folder}/{self.videoName}_temp/input_frames/'):
                start_time = time.time()
                
                
                
                    # Do some work for each iteration
                    
                try:
                    
                        
                        
                    completed_iterations = len(os.listdir(f'{self.render_folder}/{self.videoName}_temp/output_frames/'))
                    
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
    def getPreviewImage(self):
        
       
        self.imageDisplay=None
        
        if os.path.exists(f'{self.render_folder}/{self.videoName}_temp/input_frames/'):
            
            
            try:
                    files = os.listdir(f'{self.render_folder}/{self.videoName}_temp/output_frames/')
                    files.sort()
                    
                    
                    self.imageDisplay = f"{self.render_folder}/{self.videoName}_temp/output_frames/{files[-1]}"
            except:
                    self.imageDisplay = None
                    self.ui.imagePreview.clear()
                    self.ui.imagePreviewESRGAN.clear()
    
    def reportProgress(self, n):
        try:
            self.getPreviewImage()
            fp = n
            
            # fc is the total file count after interpolation
            fc = int(VideoName.return_video_frame_count(f'{self.input_file}') * self.times)
            self.fileCount = fc
            if self.i==1:
                total_input_files = len(os.listdir(f'{settings.RenderDir}/{self.videoName}_temp/input_frames/'))
                total_output_files = total_input_files * self.times 
                
                
                if self.render == 'rife':
                    self.ui.RifePB.setMaximum(total_output_files)
                    self.addLinetoLogs(f'Starting {self.times}X Render')
                    self.addLinetoLogs(f'Model: {self.ui.Rife_Model.currentText()}')
                else:
                    self.ui.ESRGANPB.setMaximum(total_output_files)
                    self.addLinetoLogs(f'Starting {self.ui.RealESRGAN_Times.currentText()[0]}X Render')
                    self.addLinetoLogs(f'Model: {self.ui.RealESRGAN_Model.currentText()}')
                self.original_fc=fc/self.times # this makes the original file count. which is the file count before interpolation
                self.i=2
            
                
            fp=int(fp)
            fc = int(fc)

            #Update GUI values
            if self.render == 'rife':
                self.ui.RifePB.setValue(fp)
                self.ui.processedPreview.setText(f'Files Processed: {fp} / {fc}')
            else:
                self.ui.ESRGANPB.setValue(fp)
                self.ui.processedPreviewESRGAN.setText(f'Files Processed: {fp} / {fc}')
            
            if self.imageDisplay != None:

                try:
                    if os.path.exists(self.imageDisplay):
                        self.ui.imageSpacerFrame.hide()
                        self.ui.imageSpacerFrameESRGAN.hide()
                        pixMap = QPixmap(self.imageDisplay)
                        
                        width = self.width()
                        height = self.height()
                        
                        width1=int(width/1.6)
                        height1=int(width1/self.aspectratio)
                        if height1 >= height/1.6:
                            
                            height1=int(height/1.6)
                            width1=int(height1/(self.videoheight/self.videowidth))
                        try:
                            if os.path.exists(self.imageDisplay):
                                pixMap = pixMap.scaled(width1,height1)
                                
                                
                                if self.render == 'rife':
                                    self.ui.imagePreview.setPixmap(pixMap) # sets image preview image
                                else:
                                    self.ui.imagePreviewESRGAN.setPixmap(pixMap)
                        except:
                            pass
                except:
                    self.ui.imageSpacerFrame.show()
                    self.ui.imageSpacerFrameESRGAN.show()
                    self.ui.imagePreview.clear()
                    self.ui.imagePreviewESRGAN.clear()
            try:
                if self.ETA != None:
                    self.ui.ETAPreview.setText(self.ETA)
                if self.i == 1 and os.path.exists(f'{self.render_folder}/{self.videoName}_temp/output_frames'):
                    self.ui.logsPreview.append(f'Starting {self.times}X Render')
                    self.i = 2
            except:
                pass
        except:
            pass
        
    def runPB(self):
        self.addLast=False
        self.i=1
        # Step 2: Create a QThread object
        self.thread = QThread()
        # Step 3: Create a worker object
       
        self.worker = workers.pb2X(self.input_file,self.render)        
        

        

        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.reportProgress)
        
        
        # Step 6: Start the thread
        
        self.thread.start()
        
        # Final resets
        
        self.worker.finished.connect(
            self.endRife
        )
       
    
        
    
        
    def settings_menu(self):
        item = self.ui.SettingsMenus.currentItem()
        if item.text() == "Video Options":
            self.ui.RenderOptionsFrame.hide()
            self.ui.VideoOptionsFrame.show()
            self.ui.GeneralOptionsFrame.hide()
        if item.text() == "Render Options":
            self.ui.RenderOptionsFrame.show()
            self.ui.VideoOptionsFrame.hide()
            self.ui.GeneralOptionsFrame.hide()
        if item.text() == "General":
            self.ui.RenderOptionsFrame.hide()
            self.ui.VideoOptionsFrame.hide()
            self.ui.GeneralOptionsFrame.show()
    
    
    def greyOutRifeTimes(self):
        if self.ui.Rife_Model.currentText() == 'Rife-V4' or self.ui.Rife_Model.currentText() == 'Rife-V4.6':
            self.ui.Rife_Times.setEnabled(True)
        else:
            self.ui.Rife_Times.setCurrentText('2X')
            self.ui.Rife_Times.setEnabled(False)
    def greyOutRealSRTimes(self):
        if self.ui.RealESRGAN_Model.currentText() == 'Default':
            self.ui.RealESRGAN_Times.setCurrentText('4X')
            self.ui.RealESRGAN_Times.setEnabled(False)
        else:
            
            self.ui.RealESRGAN_Times.setEnabled(True)
    def openFileNameDialog(self):

        self.input_file = QFileDialog.getOpenFileName(self, 'Open File', f'{homedir}',"Video files (*.mp4);;All files (*.*)")[0]
        self.videoName = VideoName.return_video_name(f'{self.input_file}')
    def openFolderDialog(self):
        
        self.output_folder = QFileDialog.getExistingDirectory(self, 'Open Folder')
        print(self.output_folder)

   

    
    def setDisableEnable(self,mode):
        self.ui.RifeStart.setDisabled(mode)
        self.ui.RealESRGANStart.setDisabled(mode)
        self.ui.Input_video_rife.setDisabled(mode) 
        self.ui.Input_video_RealESRGAN.setDisabled(mode)
        self.ui.Output_folder_rife.setDisabled(mode)
        self.ui.Output_folder_RealESRGAN.setDisabled(mode)
        self.ui.Rife_Model.setDisabled(mode)
        self.ui.RealESRGAN_Model.setDisabled(mode)
        self.ui.Rife_Times.setDisabled(mode)
        self.ui.RealESRGAN_Times.setDisabled(mode)
        self.ui.verticalTabWidget.tabBar().setDisabled(mode)
        
            
    def endRife(self): # Crashes most likely due to the fact that it is being ran in a different thread
        sleep(1)
        try:
            self.RPC.clear(pid=os.getpid())
        except:
            pass
        self.addLinetoLogs(f'Finished! Output video: {self.output_file}\n')
        self.setDisableEnable(False)
        self.ui.RifePB.setValue(self.ui.RifePB.maximum())
        self.ui.ETAPreview.setText('ETA: 00:00:00')
        self.ui.imagePreview.clear()
        self.ui.processedPreview.setText(f'Files Processed: {self.fileCount} / {self.fileCount}')
        self.ui.imageSpacerFrame.show()
        
    #The code below here is a multithreaded mess, i will fix later with proper pyqt implementation
    
    def showDialogBox(self,message,displayInfoIcon=False):
        icon = QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Info.png")
        msg = QMessageBox()
        msg.setWindowTitle(" ")
        if displayInfoIcon == True:
            msg.setIconPixmap(icon.pixmap(32, 32)) 
        msg.setText(f"{message}")
        
        msg.exec_()
    
    
    def addLinetoLogs(self,line):
        self.ui.logsPreviewESRGAN.append(f'{line}')
        self.ui.logsPreview.append(f'{line}')
    def removeLastLineInLogs(self):
        
        cursor = self.ui.logsPreview.textCursor()
        cursor.movePosition(QTextCursor.End)

        # Move the cursor to the beginning of the last line
        cursor.movePosition(QTextCursor.StartOfLine, QTextCursor.MoveAnchor)
        cursor.movePosition(QTextCursor.PreviousBlock, QTextCursor.KeepAnchor)

        # Remove the selected text (the last line)
        cursor.removeSelectedText()
        
        self.ui.logsPreview.setTextCursor(cursor)
        
if os.path.isfile(f'{thisdir}/files/settings.txt') == False:
    ManageFiles.create_folder(f'{thisdir}/files')
    ManageFiles.create_file(f'{thisdir}/files/settings.txt')
settings = Settings()


app = QtWidgets.QApplication(sys.argv)

window = MainWindow()
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor

# Force the style to be the same on all OSs:
app.setStyle("Fusion")

# Now use a palette to switch to dark colors:
palette = QPalette()
palette.setColor(QPalette.Window, QColor(53, 53, 53))
palette.setColor(QPalette.WindowText, Qt.white)
palette.setColor(QPalette.Base, QColor(25, 25, 25))
palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
palette.setColor(QPalette.ToolTipBase, Qt.black)
palette.setColor(QPalette.ToolTipText, Qt.white)
palette.setColor(QPalette.Text, Qt.white)
palette.setColor(QPalette.Button, QColor(53, 53, 53))
palette.setColor(QPalette.ButtonText, Qt.white)
palette.setColor(QPalette.Disabled, QPalette.Base, QColor(49, 49, 49))
palette.setColor(QPalette.Disabled, QPalette.Text, QColor(90, 90, 90))
palette.setColor(QPalette.Disabled, QPalette.Button, QColor(42, 42, 42))
palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(90, 90, 90))
palette.setColor(QPalette.Disabled, QPalette.Window, QColor(49, 49, 49))
palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(90, 90, 90))
palette.setColor(QPalette.BrightText, Qt.red)
palette.setColor(QPalette.Link, QColor(42, 130, 218))
palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
palette.setColor(QPalette.HighlightedText, Qt.black)
app.setPalette(palette)
sys.exit(app.exec_())
    
