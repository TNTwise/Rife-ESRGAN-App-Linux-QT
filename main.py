#!/usr/bin/python3
import os
homedir = os.path.expanduser(r"~")
import src.thisdir
import src.checks as checks
thisdir = src.thisdir.thisdir()
if os.path.exists(f'{thisdir}') == False:
    os.mkdir(f'{thisdir}')

    
import src.theme as theme
import traceback

import src.getModels.select_models as sel_mod
import src.getModels.get_models_settings
from PyQt5 import QtWidgets
import sys
from PyQt5.QtCore import QThread
import psutil
from PyQt5.QtWidgets import  QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap,QIcon
import mainwindow
import os
from threading import *
from src.settings import *
from src.return_data import *
ManageFiles.create_folder(f'{thisdir}/files/')
import src.workers as workers
import time
#import src.get_models as get_models
from time import sleep
from multiprocessing import cpu_count
from src.messages import *
import modules.Rife as rife
import modules.ESRGAN as esrgan
import modules.Waifu2X as Waifu2X
import src.onProgramStart
from src.ETA import *
from src.getLinkVideo.get_video import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QListWidget, QFileDialog, QListWidgetItem
import modules.interpolate as interpolate
import modules.upscale as upscale
from PyQt5.QtWidgets import  QVBoxLayout, QLabel,QProgressBar
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap
from src.log import log
import magic
from src.return_latest_update import *
def switch_theme(value):
    
    settings = Settings()
    settings.change_setting('Theme',f'{value}')
    theme.set_theme(app)
def setPixMap(self):
    while self.main.on==True:
        pixmap = QPixmap(f"{thisdir}/icons/Dragndrop.png")
        pixmap = pixmap.scaled(self.width(), 500, aspectRatioMode=Qt.KeepAspectRatio)
        if pixmap.isNull():
                os.system(f'rm -rf {thisdir}/icons/')
                sel_mod.install_icons()
            
                
        self.setPixmap(pixmap)
        sleep(.1)
class FileDropWidget(QLabel):
    def __init__(self, parent=None):
        super(FileDropWidget, self).__init__(parent)
        self.main = parent

        image_thread = Thread(target=lambda: setPixMap(self))
        image_thread.start()
        
        self.setAcceptDrops(True)
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                self.add_file_item(file_path)

    def add_file_item(self, file_path):
        item = QListWidgetItem(file_path)
        try:
            mime = magic.Magic(mime=True)
            filename = mime.from_file(item.text())
            if filename.find('video') != -1:
            
                # success!
                self.main.input_file = item.text()
                
                
                self.main.download_youtube_video_command = ''
                self.main.localFile = True
                self.main.videoName = VideoName.return_video_name(f'{self.main.input_file}')
                if '"' in self.main.input_file:
                    quotes(self.main)
                    self.main.input_file = ''
                else:
                    self.main.showChangeInFPS()
                    self.main.ui.logsPreview.clear()
                    self.main.addLinetoLogs(f'Input file = {item.text()}')
            else:
                not_a_video(self.main)
        except Exception as e:
            self.main.showDialogBox(e)
            traceback_info = traceback.format_exc()
            log(f'{e} {traceback_info}')
            
class MainWindow(QtWidgets.QMainWindow):
         
    def __init__(self):
        
        super(MainWindow, self).__init__()
        self.ui = mainwindow.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setMinimumSize(1000, 550)
        self.resize(1000, 550)
        self.on = True
        #print(self.ui.denoiseLevelSpinBox.value())
        try:
            self.thread2 = QThread()
        # Step 3: Create a worker object
            worker = return_latest()
            

            

            # Step 4: Move worker to the thread
            worker.moveToThread(self.thread2)
            # Step 5: Connect signals and slots
            self.thread2.started.connect(worker.run)
            worker.finished.connect(self.thread2.quit)
            worker.finished.connect(worker.deleteLater)
            self.thread2.finished.connect(self.thread2.deleteLater)
            worker.progress.connect(self.addVersionstoLogs)
            # Step 6: Start the thread
            
            self.thread2.start()
            
        except Exception as e:
            self.showDialogBox(e)
            traceback_info = traceback.format_exc()
            log(f'{e} {traceback_info}')
            print(f'{e} {traceback_info}')
        try:
            self.ui.installModelsProgressBar.setMaximum(100)
            self.localFile = True
            src.onProgramStart.onApplicationStart(self)
            self.ui.Input_video_rife_url.clicked.connect(lambda: get_linked_video(self))
            self.download_youtube_video_command = ''
            self.file_drop_widget = FileDropWidget(self)
            self.ui.imageFormLayout.addWidget(self.file_drop_widget)
            self.ui.themeCombo.setCurrentText(settings.Theme)
            self.ui.themeCombo.currentTextChanged.connect(lambda: switch_theme(self.ui.themeCombo.currentText()))
            self.ui.frameIncrementsModeCombo.setCurrentText(self.settings.FrameIncrementsMode)
            self.ui.InstallButton.clicked.connect(lambda: src.getModels.get_models_settings.run_install_models_from_settings(self))
            selFrameIncrementsMode(self)
            
        except Exception as e:
            self.showDialogBox(e)
            traceback_info = traceback.format_exc()
            log(f'{e} {traceback_info}')
        self.show()

    def restore_default_settings(self):
        with open(f'{thisdir}/files/settings.txt','w') as  f:
            pass  
        src.onProgramStart.onApplicationStart(self)
        self.ui.verticalTabWidget.setCurrentIndex(1)
        self.ui.SettingsMenus.setCurrentRow(0)
        self.ui.GeneralOptionsFrame.show()

    def addVersionstoLogs(self,n):
        self.addLinetoLogs(f'Latest Stable: {n[1]}       Latest Beta: {n[0]}')

    def changeVRAM(self):
        self.settings.change_setting('VRAM', f'{self.ui.vramAmountSpinbox.value()}')
        self.gpuMemory=self.settings.VRAM

    def setDirectories(self):
        self.models_dir=f"{thisdir}/models/"

    def switchUI(self):
        if self.ui.AICombo.currentText() == 'Rife':
            rife.modelOptions(self)

        if self.ui.AICombo.currentText() == 'RealESRGAN':
            esrgan.modelOptions(self)

        if self.ui.AICombo.currentText() == 'Waifu2X':
            Waifu2X.modelOptions(self)
    def get_pid(self,name):
        

            p = psutil.process_iter(attrs=['pid', 'name'])
            for process in p:
                if process.info['name'] == name:
                    pid = process.info['pid']
                    
                    return pid
            
    def resume_render(self):
        self.ui.RifeResume.hide() #show resume button
        
        #Thread(target=lambda: Rife(self,(self.ui.Rife_Model.currentText().lower()),2,self.input_file,self.output_folder,1)).start()
        self.ui.RifePause.show()
        
    
    def showChangeInFPS(self,localFile=True):
        
        try:
            width,height = return_data.VideoName.return_video_resolution(self.input_file)
            if int(width) > 3840 or int(height) > 2160:
                    too_large_video(self)
            if self.render == 'rife':
                
                if self.input_file != '':
                    self.times = int(self.ui.Rife_Times.currentText()[0])
                    self.ui.FPSPreview.setText(f'FPS: {(round(VideoName.return_video_framerate(self.input_file)))} -> {round(VideoName.return_video_framerate(self.input_file)*int(self.times))}')
                    
                if self.fps != None:    
                    self.ui.FPSPreview.setText(f'FPS: {round(self.fps)} -> {round(self.fps)*int(self.times)}')
            if self.render == 'esrgan':
                if self.input_file != '':
                    self.resIncrease = int(self.ui.Rife_Times.currentText()[0])
                    try:
                        if self.youtubeFile == True:
                            self.ui.FPSPreview.setText(f'RES: {self.ytVidRes} -> {int(self.ytVidRes.split("x")[0])*self.resIncrease}x{int(self.ytVidRes.split("x")[1])*self.resIncrease}')
                    except:
                        self.ui.FPSPreview.setText(f'RES: {int(VideoName.return_video_resolution(self.input_file)[0])}x{int(VideoName.return_video_resolution(self.input_file)[1])} -> {int(VideoName.return_video_resolution(self.input_file)[0])*self.resIncrease}x{int(VideoName.return_video_resolution(self.input_file)[1])*self.resIncrease}')
        except Exception as e:
            #print(e)
            pass
    
    def reportProgress(self, files_processed):
        try:
            
            
            # fc is the total file count after interpolation
            
            if self.i==1: # put every gui change that happens on start of render here
                
                fc = int(VideoName.return_video_frame_count(f'{self.input_file}') * self.times)
                self.filecount = fc
                total_input_files = fc / self.times
                total_output_files = fc
                self.ui.RifePB.setMaximum(total_output_files)
                #self.ui.QueueButton.show()
                
                Thread(target=lambda: calculateETA(self,fc)).start()
                    
                self.addLinetoLogs(f'Starting {self.ui.Rife_Times.currentText()[0]}X Render')
                self.addLinetoLogs(f'Model: {self.ui.Rife_Model.currentText()}')
            
                self.original_filecount=self.filecount/self.times # this makes the original file count. which is the file count before interpolation
                self.i=2
            
                
            fp=files_processed
            self.filecount = int(self.filecount)
            videos_rendered=0
            for i in os.listdir(f'{self.render_folder}/{self.videoName}_temp/output_frames/'):
                if 'mp4' in i:
                    videos_rendered+=1
            try:
                self.removeLastLineInLogs("Video segments created: ")
                self.addLinetoLogs(f"Video segments created: {videos_rendered}/{self.interpolation_sessions}")
            except:
                pass
            #Update GUI values
            
            self.ui.RifePB.setValue(fp)
            self.ui.processedPreview.setText(f'Files Processed: {fp} / {self.filecount}')
           
            try:
                if self.ETA != None:
                    self.ui.ETAPreview.setText(self.ETA)
                if self.i == 1 and os.path.exists(f'{self.render_folder}/{self.videoName}_temp/output_frames/'):
                    self.ui.logsPreview.append(f'Starting {self.times}X Render')
                    self.i = 2
            except Exception as e:
                #print(e)
                pass
        except Exception as e:
            #print(e)
            pass
    def runPB(self):
        self.addLast=False
        self.i=1
        self.settings = Settings()
        # Step 2: Create a QThread object
        self.thread = QThread()
        # Step 3: Create a worker object
       
        self.worker = workers.pb2X(self.input_file,self.render,self)        
        

        

        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.reportProgress)
        self.worker.image_progress.connect(self.imageViewer)
        
        # Step 6: Start the thread
        
        self.thread.start()
        
        # Final resets
        
        
       
    def imageViewer(self,step):
        if step == '1':
            self.ui.centerLabel.hide()
            self.ui.imageSpacerFrame.hide()
            try:
                img = Image.open(self.imageDisplay)
                try:
                    img.verify()
                except:
                    return
                self.pixMap = QPixmap(self.imageDisplay)
            except:
                print('Cannot open image!')
        if step == '2':
            self.pixMap = self.pixMap.scaled(self.width1,self.height1)
            self.ui.imagePreview.setPixmap(self.pixMap) # sets image preview image
        if step == '3':
            self.ui.imageSpacerFrame.show()

            self.ui.imagePreview.clear()
        
    
        
    def settings_menu(self):
        item = self.ui.SettingsMenus.currentItem()
        if item.text() == "Video Options":
            self.ui.RenderOptionsFrame.hide()
            self.ui.VideoOptionsFrame.show()
            self.ui.GeneralOptionsFrame.hide()
            self.ui.InstallModelsFrame.hide()
        if item.text() == "Render Options":
            self.ui.RenderOptionsFrame.show()
            self.ui.VideoOptionsFrame.hide()
            self.ui.GeneralOptionsFrame.hide()
            self.ui.InstallModelsFrame.hide()
        if item.text() == "General":
            self.ui.RenderOptionsFrame.hide()
            self.ui.VideoOptionsFrame.hide()
            self.ui.GeneralOptionsFrame.show()
            self.ui.InstallModelsFrame.hide()
        if item.text() == "Install Models":
            self.ui.RenderOptionsFrame.hide()
            self.ui.VideoOptionsFrame.hide()
            self.ui.GeneralOptionsFrame.hide()
            self.ui.InstallModelsFrame.show()
    
    
    def greyOutRifeTimes(self):
        if 'v4' in self.ui.Rife_Model.currentText():
            self.ui.Rife_Times.setEnabled(True)
        else:
            self.ui.Rife_Times.setCurrentText('2X')
            self.ui.Rife_Times.setEnabled(False)
    def greyOutRealSRTimes(self):
        if self.ui.AICombo.currentText() == 'RealESRGAN':
            if self.ui.Rife_Model.currentText() == 'Default':
                self.ui.Rife_Times.setCurrentText('4X')
                self.ui.Rife_Times.setEnabled(False)
            else:
                
                self.ui.Rife_Times.setEnabled(True)
        if self.ui.AICombo.currentText() == 'Waifu2X':
            if self.ui.Rife_Model.currentText() != 'cunet':
                self.ui.Rife_Times.setCurrentText('2X')
                self.ui.Rife_Times.setEnabled(False)
            else:
                
                self.ui.Rife_Times.setEnabled(True)
    def openFileNameDialog(self):

        self.input_file = QFileDialog.getOpenFileName(self, 'Open File', f'{homedir}',"Video files (*.mp4);;All files (*.*)")[0]
        print(self.input_file)
        try:
            mime = magic.Magic(mime=True)
            filename = mime.from_file(self.input_file)
            if filename.find('video') != -1:
            
                # success!
                
                
                
                self.download_youtube_video_command = ''
                self.localFile = True
                self.videoName = VideoName.return_video_name(f'{self.input_file}')
                if '"' in self.input_file:
                    quotes(self)
                    self.input_file = ''
                else:
                    self.showChangeInFPS()
                    self.ui.logsPreview.clear()
                    self.addLinetoLogs(f'Input file = {self.input_file}')
            else:
                not_a_video(self)
        except Exception as e:
            self.showDialogBox(e)
            traceback_info = traceback.format_exc()
            log(f'{e} {traceback_info}')
    def openFolderDialog(self):
        
        self.output_folder = QFileDialog.getExistingDirectory(self, 'Open Folder')

   
    def pause_render(self):
        # Why was this line here??
            self.paused = True
            self.ui.RifePause.hide()
            
                
                
                
            os.system(f'kill -9 {self.get_pid("rife-ncnn-vulkan")}')
            os.system(f'kill -9 {self.get_pid("realesrgan-ncnn-vulkan")}')
            sleep(0.1)
            files_to_delete = len(os.listdir(f'{settings.RenderDir}/{self.videoName}_temp/output_frames/')) / self.times
            for i in range(int(files_to_delete)):
                i = str(i).zfill(8)
                os.system(f'rm -rf "{settings.RenderDir}/{self.videoName}_temp/input_frames/{i}.png"')
            self.ui.RifeResume.show() #show resume button
                #This function adds a zero to the original frames, so it wont overwrite the old ones
    def setDisableEnable(self,mode):
        self.ui.AICombo.setDisabled(mode)
        self.ui.RifeStart.setDisabled(mode)
        self.ui.Input_video_rife.setDisabled(mode) 
        self.ui.Input_video_rife_url.setDisabled(mode) 
        self.ui.Output_folder_rife.setDisabled(mode)
        self.ui.Rife_Model.setDisabled(mode)
        self.ui.Rife_Times.setDisabled(True)
        if 'v4' in self.ui.Rife_Model.currentText().lower():
            self.ui.Rife_Times.setDisabled(mode)
        
        self.ui.verticalTabWidget.tabBar().setDisabled(mode)
        self.ui.denoiseLevelSpinBox.setDisabled(mode)
        self.ui.InstallModelsFrame.setDisabled(mode)
        self.ui.SettingsMenus.setDisabled(mode)
    def endRife(self): # Crashes most likely due to the fact that it is being ran in a different thread
        if len(self.QueueList) == 0:
            self.ui.QueueListWidget.hide()
            try:
                self.RPC.clear(pid=os.getpid())
            except:
                pass
            self.ui.RifePause.hide()
            self.ui.RifeResume.hide()
            self.ui.QueueButton.hide()
            self.ui.centerLabel.show()
            self.addLinetoLogs(f'Finished! Output video: {self.output_file}\n\n')
            self.setDisableEnable(False)
            self.ui.RifePB.setValue(self.ui.RifePB.maximum())
            self.ui.ETAPreview.setText('ETA: 00:00:00')
            self.ui.imagePreview.clear()
            self.ui.processedPreview.setText(f'Files Processed: {self.filecount} / {self.filecount}')
            self.ui.imageSpacerFrame.show()
            self.greyOutRealSRTimes()
        if len(self.QueueList) > 0:
            self.input_file = self.QueueList[0]
            del self.QueueList[0]
            self.ui.QueueListWidget.takeItem(0)
            if self.render == 'rife':
                interpolate.start_interpolation(self,'rife-ncnn-vulkan')
            if self.render == 'esrgan' and self.ui.AICombo.currentText() != 'Waifu2x':
                upscale.start_upscale(self,'realesrgan-ncnn-vulkan')
            if self.ui.AICombo.currentText() == 'Waifu2x':
                upscale.start_upscale(self,'waifu2x-ncnn-vulkan')
        
        
        

        
    
    def showDialogBox(self,message,displayInfoIcon=False):
        icon = QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Info.png")
        msg = QMessageBox()
        msg.setWindowTitle(" ")
        if displayInfoIcon == True:
            msg.setIconPixmap(icon.pixmap(32, 32)) 
        msg.setText(f"{message}")
        
        msg.exec_()

    def showQuestionBox(self,message):
        reply = QMessageBox.question(self, '', f'{message}', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            return True
        else:
            return False

    def closeEvent(self, event):
        if self.input_file != '':
            
            reply = QMessageBox.question(
                
                self,
                "Confirmation",
                
                "Are you sure you want to exit?\n(The current render will be killed)",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
                
            )
            
        else:
            reply = QMessageBox.question(
                
                self,
                "Confirmation",
                
                "Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
        if reply == QMessageBox.Yes:
            self.on = False
            event.accept()
            if self.input_file != '':
                try:
                    os.system(f'rm -rf "{settings.RenderDir}/{self.videoName}_temp/"')
                    os.system(f'rm -rf "{thisdir}/{self.videoName}"')
                    for i in os.listdir(f'{thisdir}'):
                        mime = magic.Magic(mime=True)
                        filename = mime.from_file(f'{thisdir}/{i}')
                        if filename.find('video') != -1:
                        
                            os.system(f'rm -rf "{thisdir}/{i}"')
                        if '.mp4' in i:
                            os.system(f'rm -rf "{thisdir}/{i}"')
                except Exception as e:
                    log(str(e))
                
                self.ffmpeg.terminate()
                self.renderAI.terminate()

                try:
                    os.system(f'rm -rf "{settings.RenderDir}/{self.videoName}_temp/"')
                except:
                    pass
                #os.system(f'kill -9 {os.getpid()}')
                exit()
        else:
            event.ignore()
    
    def addLinetoLogs(self,line,remove_text=''):
        if line != 'REMOVE_LAST_LINE' or remove_text != '':
            self.ui.logsPreview.append(f'{line}')
        else:
            self.removeLastLineInLogs(remove_text)
            
    def update_last_line(self,new_line_text):
        # Assuming line number is 2 (index 1) - replace with the desired line number
        line_number = 1

        cursor = self.ui.logsPreview.textCursor()
        cursor.movePosition(cursor.Start)
        for _ in range(line_number):
            cursor.movePosition(cursor.Down, cursor.KeepAnchor)
        cursor.removeSelectedText()
        cursor.insertText(new_line_text)

    def removeLastLineInLogs(self,text_in_line):#takes in text in line and removes every line that has that specific text.
        text = self.ui.logsPreview.toPlainText().split('\n')
        text1=[]
        for i in text:
            if i != ' 'or i != '' :
                
                if len(i) > 3:
                    text1.append(i)
        text = text1
        display_text = ''
        for i in text:
            if text_in_line not in i:
                display_text+=f'{i}'
                if i != ' ':
                    display_text+='\n'
            else:
                #print(i)
                pass
        self.ui.logsPreview.clear()
        
        self.ui.logsPreview.setText(display_text)
        scroll_bar = self.ui.logsPreview.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
try:
    if os.path.isfile(f'{thisdir}/files/settings.txt') == False:
        ManageFiles.create_folder(f'{thisdir}/files')
        ManageFiles.create_file(f'{thisdir}/files/settings.txt')
    settings = Settings()
except Exception as e:
    
    traceback_info = traceback.format_exc()
    log(f'{e} {traceback_info}')

try:
    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QPalette, QColor

    # Force the style to be the same on all OSs:
    theme.set_theme(app)
    log('Program Started')
    
    sys.exit(app.exec_())
except Exception as e:
    traceback_info = traceback.format_exc()
    log(f'{e} {traceback_info}')

