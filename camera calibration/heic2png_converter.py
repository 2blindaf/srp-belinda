import glob
from PIL import Image
import pillow_heif

def heic2png(heicfolder, pngfolder):
    images = glob.glob(f'{heicfolder}/*.heic') # or use os.scandir instead for the same result
    for fileName in images:
        heif_file = pillow_heif.read_heif(fileName)
        image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, 'raw')
        new_fileName = pngfolder + fileName[len(heicfolder):-4] + 'png'
        image.save(new_fileName, format('png'))
        print(f'{new_fileName} has been created!')

heic2png('heic_images', 'png_images')