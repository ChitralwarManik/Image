#pip install ultralytics
import zipfile
import os
import time

from ultralytics import YOLO
import sys
# Load a model
import sys, os
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))


def MLImageYolo(model_Path, image_Path):
    try:

        # "D:\ML MODEL\ImageToPickelEXE##im$$D:\TEST_UAT\PII PCI combined_Agent\MLImageScan\Kavita pan.jpg"
        # names = ['AadharCard', 'AmericanExpress', 'DinersClub', 'Discover', 'JCB', 'Maestro', 'MasterCard', 'PANCard', 'Rupay', 'UnionPay', 'VISA', 'VoterId']
        names = ['Aadhaar (IND)', 'American Express', 'CanaraBankLOGO', 'Diners Club', 'Discover', 'Driving License',
                 'Passport (India)', 'JCB', 'Maestro', 'Master Card', 'NINO (UK)', 'NRIC (Singapore)',
                 'National ID (Saudi Arabia)', 'National ID (UAE)', 'PAN (IND)', 'RuPay', 'SSN (US)', 'Union Pay',
                 'VISA', 'Voter ID (IND)']
        # for i in range(100):
        model = YOLO(model_Path)
        # time.sleep(2)

        result = model(image_Path)
        boxes = result[0].boxes
        Clsses = []
        for ele in boxes.cls:
            Clsses.append(names[int(ele)])

        confidences = []
        for con in boxes.conf:
            confidences.append(float(con))

        res = dict(zip(confidences, Clsses))
        if len(res) == 0:
            sys.stdout.write("$*$" + "Class Not Found" + "$*$")
        else:
            for key, value in res.items():
                sys.stdout.write("$*$" + str(value) + ':' + str(key) + "$*$")

    except Exception as e:
        print("Error invalid file format: " + e.__str__())


if __name__ == "__main__":
    configString = sys.argv[1]

    #InstalledLocation##im$$imageLocation

    configs = configString.split("##im$$")

    modelPath = os.path.join(configs[0],"All21PII.pt")

    FilePath = configs[1]
    target_dir = "output"
    password = "Sisa!1@2#3$4"
    if FilePath.endswith(".zip"):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with zipfile.ZipFile(FilePath, 'r') as zip_ref:
            # Extract all files to the target directory
            zip_ref.extractall(target_dir,pwd=password.encode())
        for images
    else:
        MLImageYolo(modelPath,FilePath)


