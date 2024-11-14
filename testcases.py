from src.ModelHandler import ncnnInterpolateModels, pytorchInterpolateModels, ncnnUpscaleModels, pytorchUpscaleModels, tensorrtInterpolateModels, tensorrtUpscaleModels
import requests
import os
import tarfile

MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")

def removeFile(file):
    try:
        os.remove(file)
    except Exception:
        print("Failed to remove file!")

def extractTarGZ(file):
    """
    Extracts a tar gz in the same directory as the tar file and deleted it after extraction.
    """
    origCWD = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(file))
    os.chdir(dir_path)
    print("Extracting: " + file)
    tar = tarfile.open(file, "r:gz")
    tar.extractall()
    tar.close()
    removeFile(file)
    os.chdir(origCWD)


def download_file(url, download_path):
    """
    Downloads a file from the given URL and saves it to the specified path.
    
    :param url: The URL of the file to download.
    :param download_path: The path where the file should be saved.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the request was successful
        
        # Open the file in write-binary mode and write the content
        with open(download_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print(f"File downloaded successfully: {download_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download file: {e}")
    

def downloadModel(modelFile, downloadModelPath: str = None):
        url = (
            "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/"
            + modelFile
        )
        model_on_filesystem = os.path.join(downloadModelPath,modelFile)
        if os.path.isfile(model_on_filesystem) or os.path.exists(model_on_filesystem.replace(".tar.gz","")):
            return
        download_file(url, model_on_filesystem)
        print("Done")
        if "tar.gz" in modelFile:
            print("Extracting File")
            extractTarGZ(model_on_filesystem)


def downloadModelsFromModelList(model_list: list):
    for model in model_list:
        downloadModel(modelFile=model_list[model][1], downloadModelPath=MODEL_PATH)
def downloadModels():
    downloadModelsFromModelList(ncnnInterpolateModels)
    downloadModelsFromModelList(pytorchInterpolateModels)
    downloadModelsFromModelList(ncnnUpscaleModels)
    downloadModelsFromModelList(pytorchUpscaleModels)

def main():
    downloadModels()
    #render video

if __name__ == '__main__':
    main()