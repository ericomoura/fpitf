import cv2
import color_transfer
import numpy as np
import skimage.exposure as skitra


# Merges two images ignoring transparent pixels
def mergeImages(bg, obj, objAlpha=0.5):
    width, height, channels = obj.shape
    if(channels < 4):
        raise Exception("Object not transparent!")
    result = np.ones((width, height, 3), np.uint8)

    for column in range(width-1):
        for row in range(height-1):
            if(obj[column][row][3] == 0):  # Transparent pixel
                result[column][row][0:3] = bg[column][row][0:3]
            else:  # Non transparent pixel
                result[column][row][0:3] = obj[column][row][0:3]*objAlpha + bg[column][row][0:3]*(1-objAlpha)

    return result

#Applies a quantization technique reducing the number of shades to the specified number
def quantize(image, shades):
    height, width, channels = image.shape
    newImage = image.copy()

    if(shades < 1): #Can't have less than 1 shade, returns a black image
        newImage = np.zeros([height, width, channels])
    else:
        for h in range(height):
            for w in range(width):
                b, g, r = image[h][w]
                newB = round(b/(255/shades)) * (255/shades)
                newG = round(g/(255/shades)) * (255/shades)
                newR = round(r/(255/shades)) * (255/shades)
                newImage[h][w] = [newB, newG, newR]

    return newImage

def main():
    scene = "./images/scene1/"
    bgImage = cv2.imread(scene+"bg.png", cv2.IMREAD_COLOR)
    obj = cv2.imread(scene+"obj1.png", cv2.IMREAD_UNCHANGED)
    objQuant = quantize(obj[:,:,0:3], 4)
    objQuant = cv2.cvtColor(objQuant, cv2.COLOR_RGB2RGBA)
    objQuant[:,:,3] = obj[:,:,3]

    out = mergeImages(bgImage, objQuant, 0.75)

    obj = cv2.imread(scene+"obj2.png", cv2.IMREAD_UNCHANGED)
    objQuant = quantize(obj[:,:,0:3], 4)
    objQuant = cv2.cvtColor(objQuant, cv2.COLOR_RGB2RGBA)
    objQuant[:,:,3] = obj[:,:,3]

    out = mergeImages(out, objQuant, 0.75)

    cv2.imshow('Merge quantized', out)
    cv2.imwrite(scene+'scene1answer.png', out)
    debevec = cv2.createMergeDebevec()
    merged = debevec.process([bgImage, out], np.array([0.15, 0.15], dtype=np.float32))

    tonemapper = cv2.createTonemapReinhard(0.5, 0, 0, 0)  #Gamma, intensity, light_adapt, color_adapt
    tonemapped = tonemapper.process(merged)
    cv2.imshow("Merge and mapped reinhard quantized", tonemapped)
    out = cv2.convertScaleAbs(tonemapped, alpha=(255.0))
    out = out.astype('uint8')
    cv2.imwrite(scene+'scene1.png', out)



if __name__ == "__main__":
    main()
    cv2.waitKey(0)