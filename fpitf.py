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
    out = mergeImages(bgImage, out)
    cv2.imshow('Color transfer and merge', out)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2YCR_CB)
    y, cb, cr = cv2.split(out)
    y = cv2.equalizeHist(y)
    out = cv2.merge((y,cb,cr))
    out = cv2.cvtColor(out, cv2.COLOR_YCR_CB2BGR)
    cv2.imshow("Color transfer and merge equalized", out)

    out2 = mergeImages(bgImage, obj, 0.5)
    cv2.imshow('Merge', out2)
    out2 = cv2.cvtColor(out2, cv2.COLOR_BGR2YCR_CB)
    y, cb, cr = cv2.split(out2)
    y = cv2.equalizeHist(y)
    out2 = cv2.merge((y,cb,cr))
    out2 = cv2.cvtColor(out2, cv2.COLOR_YCR_CB2BGR)
    cv2.imshow("Merge equalized", out2)

    objMatched = hist_match(obj, bgImage)
    # cv2.imshow("Histogram matching", objMatched)

    objMatched = cv2.cvtColor(objMatched, cv2.COLOR_RGB2RGBA)
    objMatched[:,:,3] = obj[:,:,3]
    objMatched = mergeImages(bgImage, objMatched)
    cv2.imshow("Histogram matching and merge", objMatched)
    objMatched = cv2.cvtColor(objMatched, cv2.COLOR_BGR2YCR_CB)
    y, cb, cr = cv2.split(objMatched)
    y = cv2.equalizeHist(y)
    objMatched = cv2.merge((y,cb,cr))
    objMatched = cv2.cvtColor(objMatched, cv2.COLOR_YCR_CB2BGR)
    cv2.imshow("Histogram matching and merge equalized", objMatched)

    skimageMatched = obj[:,:,0:3]
    skimageMatched = skitra.match_histograms(skimageMatched, bgImage, multichannel=True)
    # cv2.imshow("Skimage matching", skimageMatched)

    skimageMatched = cv2.cvtColor(skimageMatched, cv2.COLOR_RGB2RGBA)
    skimageMatched[:,:,3] = obj[:,:,3]
    skimageMatched = mergeImages(bgImage, skimageMatched, 0.75)
    cv2.imshow("Skimage matching and merge", skimageMatched)
    skimageMatched = cv2.cvtColor(skimageMatched, cv2.COLOR_BGR2YCR_CB)
    y, cb, cr = cv2.split(skimageMatched)
    y = cv2.equalizeHist(y)
    skimageMatched = cv2.merge((y,cb,cr))
    skimageMatched = cv2.cvtColor(skimageMatched, cv2.COLOR_YCR_CB2BGR)
    cv2.imshow("Skimage matching and merge equalized", skimageMatched)





if __name__ == "__main__":
    main()
    cv2.waitKey(0)