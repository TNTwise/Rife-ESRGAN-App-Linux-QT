
class BorderDetect:
    def __init__(self, inputFile):
        self.inputFile = inputFile

    def processBorders(self):
        command = [
            f"{FFMPEG_PATH}",
            "-i",
            f"{self.inputFile}",
            "-vf",
            "cropdetect",
            "-f",
            "null",
            "-",
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True,
        )
        output = process.communicate()
        return output

    def processOutput(self, output):
        borders = []
        for line in output[1].split("\n"):
            if "crop=" in line:
                crop_value = line.split("crop=")[1].split(" ")[0]
                borders.append(crop_value)

        if borders:

            def parse_crop(crop_str):
                # Expected format: "width:height:x:y"
                try:
                    width, height, x, y = map(int, crop_str.split(":"))
                    if width <= 0 or height <= 0:
                        log(f"Invalid crop dimensions: {crop_str}")
                        return None
                    return width, height, x, y
                except ValueError:
                    log(f"Invalid crop format: {crop_str}")
                    return None

            # Parse all crop values and filter out any invalid entries
            parsed_crops = [parse_crop(crop) for crop in borders]
            parsed_crops = [crop for crop in parsed_crops if crop is not None]

            if not parsed_crops:
                log("No valid crop values found.")
                return None

            # Determine the least cropped crop (i.e., largest area)
            least_cropped = max(parsed_crops, key=lambda dims: dims[0] * dims[1])
            least_cropped_str = f"{least_cropped[0]}:{least_cropped[1]}:{least_cropped[2]}:{least_cropped[3]}"

            return least_cropped_str

        return None

    def getBorders(self):
        output = self.processBorders()
        output = self.processOutput(output)
        width, height, borderX, borderY = map(int, output.split(":"))
        return width, height, borderX, borderY