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


def main():
    bgImage = cv2.imread('./images/cliff1.jpg', cv2.IMREAD_COLOR)
    obj = cv2.imread('./images/obj1.png', cv2.IMREAD_UNCHANGED)
    # cv2.imshow('Background', bgImage)
    # cv2.imshow('Object', obj)

    out = color_transfer.color_transfer(bgImage, obj)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2RGBA)
    out[:,:,3] = obj[:,:,3]
    out = mergeImages(bgImage, out, 1)
    cv2.imshow('Color transfer and merge', out)
    debevec = cv2.createMergeDebevec()
    merged = debevec.process([bgImage, out], np.array([0.15, 0.15], dtype=np.float32))

    tonemapper = cv2.createTonemapReinhard(0.5, 1, 0, 0)  #Gamma, intensity, light_adapt, color_adapt
    tonemapped = tonemapper.process(merged)
    cv2.imshow("Color transfer and merge and mapped reinhard", tonemapped)

    ######

    objGray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
    out = np.ones((338, 600, 4), np.uint8)
    out[:,:,0] = objGray[:,:]
    out[:,:,1] = objGray[:,:]
    out[:,:,2] = objGray[:,:]
    out[:,:,3] = obj[:,:,3]
    objGray = out

    out = color_transfer.color_transfer(bgImage, objGray)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2RGBA)
    out[:,:,3] = obj[:,:,3]
    out = mergeImages(bgImage, out, 1)
    cv2.imshow('Gray color transfer and merge', out)
    debevec = cv2.createMergeDebevec()
    merged = debevec.process([bgImage, out], np.array([0.15, 0.15], dtype=np.float32))

    tonemapper = cv2.createTonemapReinhard(0.5, 1, 0, 0)  #Gamma, intensity, light_adapt, color_adapt
    tonemapped = tonemapper.process(merged)
    cv2.imshow("Gray color transfer and merge and mapped reinhard", tonemapped)





if __name__ == "__main__":
    main()
    cv2.waitKey(0)