from src.settings import *
from src.messages import *
import src.queue.queue as queue
import multiprocessing
import src.checks as checks
def onApplicationStart(self):
    import os
    
    #this is kind of a mess
    import modules.Rife as rife
    import modules.ESRGAN as esrgan
    settings = Settings()
    from PyQt5.QtGui import QIntValidator, QIcon
    import src.thisdir
    thisdir = src.thisdir.thisdir()
    self.ui.AICombo.clear() # needs to be in this order, before SwitchUI is called
    for i in os.listdir(settings.ModelDir):
                if i == 'rife':
                    self.ui.AICombo.addItem('Rife')
                
                if i == 'realesrgan':
                    self.ui.AICombo.addItem('RealESRGAN')
                if i == 'waifu2x':
                    self.ui.AICombo.addItem('Waifu2X')
                
    self.input_file = ''
    
    self.setWindowIcon(QIcon(f'{thisdir}/icons/logo v1.png'))
    
    self.settings = Settings()
    self.gpuMemory=self.settings.VRAM
    
    self.switchUI()
    if self.gpuMemory != 'None':
        
        self.ui.vramAmountSpinbox.setValue(int(self.gpuMemory))
        
    else:
        if self.settings.VRAM == 'None':
            cannot_detect_vram(self)
        self.ui.vramAmountSpinbox.setValue(1)
    models_installed = checks.check_for_individual_models()
    for i in models_installed:
                if 'Rife' == i:
                    self.ui.RifeCheckBox.setChecked(True)
                if 'RealESRGAN' == i:
                    self.ui.RealESRGANCheckBox.setChecked(True)
                if 'RealCUGAN' == i:
                    self.ui.RealCUGANCheckBox.setChecked(True)
                if 'Waifu2X' == i:
                    self.ui.Waifu2xCheckBox.setChecked(True)
                if 'Cain' == i:
                    self.ui.CainCheckBox.setChecked(True)
    
    
    self.ui.RifeSettings.clicked.connect(lambda: src.getModels.get_models_settings.get_rife(self))
    self.ui.installModelsProgressBar.setMaximum(100)
        
    self.ui.vramAmountSpinbox.setMinimum(1)
    
    #Define Variables
    self.input_file = ''
    self.output_folder = ''
    self.videoQuality = settings.videoQuality
    self.encoder = settings.Encoder
    if os.path.exists(f"{settings.RenderDir}") == False:
        settings.change_setting('RenderDir',f'{thisdir}')
    self.render_folder = settings.RenderDir
    self.ui.sceneChangeSensativityButton.setIcon(QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Help.png"))
    self.ui.encoderHelpButton.setIcon(QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Help.png"))
    self.ui.imageHelpButton.setIcon(QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Help.png"))
    self.ui.vramAmountHelpButton.setIcon(QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Help.png"))
    self.ui.renderTypeHelpButton.setIcon(QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Help.png"))
    self.ui.frameIncrementHelp.setIcon(QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Help.png"))
    self.ui.Rife_Times.currentIndexChanged.connect(self.showChangeInFPS)
    self.ui.vramAmountSpinbox.valueChanged.connect(self.changeVRAM)
    self.ui.AICombo.currentIndexChanged.connect(self.switchUI)
    self.ui.InstallModelsFrame.hide()
    self.ui.RifeResume.hide()
    self.ui.RifePause.hide()
    self.ui.DiscordRPCBox.stateChanged.connect(lambda: changeDiscordRPC(self))
    if settings.DiscordRPC == 'Enabled':
        self.ui.DiscordRPCBox.setChecked(True)
        
    else:
        self.ui.DiscordRPCBox.setChecked(False)
    os.system('ln -sf {app/com.discordapp.Discord,$XDG_RUNTIME_DIR}/discord-ipc-0') #Enables discord RPC on flatpak
    onlyInt = QIntValidator()
    onlyInt.setRange(0, 9)
    self.ui.sceneChangeLineEdit.setValidator(onlyInt)
    self.ui.sceneChangeLineEdit.textChanged.connect(lambda: changeSceneDetection(self))
    self.ui.sceneChangeLineEdit.setText(settings.SceneChangeDetection[2])
    if self.encoder == '264':
        self.ui.EncoderCombo.setCurrentIndex(0)
    if self.encoder == '265':
        self.ui.EncoderCombo.setCurrentIndex(1)
    if self.videoQuality == '10':
        self.ui.VidQualityCombo.setCurrentText('Lossless')
    if self.videoQuality == '14':
        self.ui.VidQualityCombo.setCurrentText('Very High')
    if self.videoQuality == '18':
        self.ui.VidQualityCombo.setCurrentText('High')
    if self.videoQuality == '20':
        self.ui.VidQualityCombo.setCurrentText('Medium')
    if self.videoQuality == '22':
        self.ui.VidQualityCombo.setCurrentText('Low')
    if self.settings.RenderType == 'Classic':
        self.ui.renderTypeCombo.setCurrentIndex(0)
    elif self.settings.RenderType == 'Optimized':
        self.ui.renderTypeCombo.setCurrentIndex(1)
    else:
        self.ui.renderTypeCombo.setCurrentIndex(2)
    self.ui.frameIncrementsModeCombo.currentTextChanged.connect(lambda: selFrameIncrementsMode(self))
    self.ui.sceneChangeDetectionCheckBox.stateChanged.connect(lambda: selSceneDetectionMode(self))
    if settings.SceneChangeDetectionMode == 'Enabled':
        self.ui.sceneChangeDetectionCheckBox.setChecked(True)
        self.ui.label_3.show()
        self.ui.sceneChangeSensativityButton.show()
        self.ui.sceneChangeLineEdit.show()
    else:
        self.ui.sceneChangeDetectionCheckBox.setChecked(False)
        self.ui.label_3.hide()
        self.ui.sceneChangeSensativityButton.hide()
        self.ui.sceneChangeLineEdit.hide()
    self.ui.renderTypeHelpButton.clicked.connect(lambda: render_help(self))
    self.ui.Rife_Model.currentIndexChanged.connect(self.greyOutRifeTimes)
    self.ui.OutputDirectoryLabel.setText(settings.OutputDir)
    self.ui.frameIncrementSpinBox.setValue(settings.FrameIncrements)
    self.ui.frameIncrementSpinBox.valueChanged.connect(lambda: selFrameIncrements(self.ui.frameIncrementSpinBox.value()))
    #link buttons
    self.ui.frameIncrementHelp.clicked.connect(lambda: frame_increments_help(self))
    self.ui.SettingsMenus.clicked.connect(self.settings_menu)
    self.ui.OutputDirectoryButton.clicked.connect(lambda: selOutputDir(self))
    self.ui.resetSettingsButton.clicked.connect(self.restore_default_settings)
    self.ui.vramAmountHelpButton.clicked.connect(lambda: vram_help(self))
    self.ui.RifePause.clicked.connect(self.pause_render)
    self.ui.RifeResume.clicked.connect(self.resume_render)
    self.ui.sceneChangeSensativityButton.clicked.connect(lambda: show_scene_change_help(self))
    self.ui.encoderHelpButton.clicked.connect(lambda:  encoder_help(self))
    self.ui.renderTypeCombo.currentIndexChanged.connect(lambda: selRenderType(self))
            
    self.ui.RenderPathLabel.setText(f"{settings.RenderDir}")
    self.ui.RenderDirButton.clicked.connect(lambda: selRenderDir(self))
    
    self.ui.Input_video_rife.clicked.connect(self.openFileNameDialog)
    self.ui.Output_folder_rife.clicked.connect(self.openFolderDialog)
    self.ui.VideoOptionsFrame.hide()
    self.ui.RenderOptionsFrame.hide()
    self.ui.GeneralOptionsFrame.hide()
   
    
    self.ui.EncoderCombo.currentIndexChanged.connect(lambda: selEncoder(self))
    #apparently adding multiple currentindexchanged causes a memory leak unless i sleep, idk why it does this but im kinda dumb
    
    self.ui.VidQualityCombo.currentIndexChanged.connect(lambda: selVidQuality(self))
    self.ui.QueueButton.clicked.connect(lambda: queue.addToQueue(self))
    self.ui.QueueButton.hide()
    self.ui.QueueListWidget.hide()
    self.QueueList=[]
    self.setDirectories()
    self.ui.imageComboBox.setCurrentText(f'{settings.Image_Type}')
    self.ui.imageHelpButton.clicked.connect(lambda: image_help(self))
    self.ui.imageComboBox.currentIndexChanged.connect(lambda: settings.change_setting('Image_Type', f'{self.ui.imageComboBox.currentText()}'))
    if self.gpuMemory == None:
        cannot_detect_vram(self)
    else:
        pass
    if int(HardwareInfo.get_video_memory_linux()) < 4:
        not_enough_vram(self)
    # list every model downloaded, and add them to the list
    self.ui.SettingsMenus.setCurrentRow(0)
    self.ui.GeneralOptionsFrame.show()

def list_model_downloaded(self):
        settings = Settings()
        model_filepaths = ([x[0] for x in os.walk(f'{settings.ModelDir}/rife/')])
        models = []
        for model_filepath in model_filepaths:
            if 'rife' in os.path.basename(model_filepath):
                models.append(os.path.basename(model_filepath))
        
        models.sort()
        for model in models:

            
            
            self.ui.Rife_Model.addItem(f'{model}')#Adds model to GUI.
            if model == 'rife-v4.6':
                self.ui.Rife_Model.setCurrentText(f'{model}')