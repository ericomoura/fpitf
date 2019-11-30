import cv2
import color_transfer
import numpy as np
import skimage.exposure as skitra



def find_nearest_above(my_array, target):
    diff = my_array - target
    mask = np.ma.less_equal(diff, -1)
    # We need to mask the negative differences
    # since we are looking for values above
    if np.all(mask):
        c = np.abs(diff).argmin()
        return c # returns min index of the nearest if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()
def hist_match(original, specified):
 
    oldshape = original.shape
    original = original.ravel()
    specified = specified.ravel()
 
    # get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(original, return_inverse=True,return_counts=True)
    t_values, t_counts = np.unique(specified, return_counts=True)
 
    # Calculate s_k for original image
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    
    # Calculate s_k for specified image
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
 
    # Round the values
    sour = np.around(s_quantiles*255)
    temp = np.around(t_quantiles*255)
    
    # Map the rounded values
    b=[]
    for data in sour[:]:
        b.append(find_nearest_above(temp,data))
    b= np.array(b,dtype='uint8')
 
    return b[bin_idx].reshape(oldshape)

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
    bgImage = cv2.imread('./images/cliff1.jpg', cv2.IMREAD_COLOR)
    obj = cv2.imread('./images/obj1.png', cv2.IMREAD_UNCHANGED)

    out = mergeImages(bgImage, obj, 0.75)
    cv2.imshow('Merge', out)
    debevec = cv2.createMergeDebevec()
    merged = debevec.process([bgImage, out], np.array([0.15, 0.15], dtype=np.float32))

    tonemapper = cv2.createTonemapReinhard(0.5, 0, 0, 0)  #Gamma, intensity, light_adapt, color_adapt
    tonemapped = tonemapper.process(merged)
    cv2.imshow("Merge and mapped reinhard", tonemapped)

    #####

    objQuant = quantize(obj[:,:,0:3], 4)
    objQuant = cv2.cvtColor(objQuant, cv2.COLOR_RGB2RGBA)
    objQuant[:,:,3] = obj[:,:,3]

    out = mergeImages(bgImage, objQuant, 0.75)
    cv2.imshow('Merge quantized', out)
    debevec = cv2.createMergeDebevec()
    merged = debevec.process([bgImage, out], np.array([0.15, 0.15], dtype=np.float32))

    tonemapper = cv2.createTonemapReinhard(0.5, 0, 0, 0)  #Gamma, intensity, light_adapt, color_adapt
    tonemapped = tonemapper.process(merged)
    cv2.imshow("Merge and mapped reinhard quantized", tonemapped)




if __name__ == "__main__":
    main()
    cv2.waitKey(0)