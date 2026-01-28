import cv2
import numpy as np
import math

def main():
    
    # Read image (BGR format) 
    image_colour = cv2.imread("imgCredit.jpeg")
    
    if image_colour is None:
        raise FileNotFoundError("Error: Image not found")
#######################COLOUR DETECTIONS#############################
    
    # Convert BGR → HSV
    hsv = cv2.cvtColor(image_colour, cv2.COLOR_BGR2HSV)

    # Define blue color range (HSV)
    lower_blue = np.array([80, 40, 40])
    upper_blue = np.array([145, 255, 255])

    # Create mask
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
  
    if np.any(blue_mask > 0):
        colour='Silver'
        print(colour)

    
#######################CREDIT DETECTIONS#############################
    

    #######################Black Mask########################################################

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([135, 135, 135])
    
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    #Thresholding for clean binary image to get sharp edges
    _, black_binary = cv2.threshold(black_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    #Opening for noise removing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    black_opening = cv2.morphologyEx(black_binary, cv2.MORPH_OPEN, kernel)
    
    combined_mask = cv2.bitwise_or(blue_mask, black_opening)
    
    
    black_contours, hierarchy = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
   
    black_cnt = max(black_contours, key=cv2.contourArea)
    # minAreaRect gives (center(x,y), (width, height), angle)
    black_rect = cv2.minAreaRect(black_cnt)
    black_box = cv2.boxPoints(black_rect)              # 4 corner points (float)
    black_box = np.rint(black_box).astype(np.int32)                     # convert to int
    # cv2.drawContours(image_colour, [black_box], 0, (0, 0, 0), 2)  # black
    
    
    ####################Find the blue credits########################
    # Extract blue regions for 
    blue_detected = cv2.bitwise_and(image_colour, image_colour, mask=blue_mask)
  
    # Create grayscale image (optional)
    image_gray = cv2.cvtColor(blue_detected, cv2.COLOR_BGR2GRAY)
    
    #Thresholding
    _, binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #Opening for noise removing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    #Contours finding
    contours,hierarchy = cv2.findContours(opening, 1, 2)
       
    #centroids of the blue credits
    cx_array=[]
    cy_array=[]
    contours_area=[]
    contour_width=[]
    
    for cnt in contours:
        # minAreaRect gives (center(x,y), (width, height), angle)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)              # 4 corner points (float)
        box = np.rint(box).astype(np.int32)                     # convert to int
        cv2.drawContours(image_colour, [box], 0, (0, 0, 255), 2)  # red
        
        (center_x, center_y), (w, h), angle = rect
        contour_width.append(w)
        #Draw centers
        M = cv2.moments(cnt)

        if M["m00"] == 0:   # avoid division by zero
                continue

        # Compute centroid
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cx_array.append(cx)
        cy_array.append(cy)
        cv2.circle(image_colour, (cx,cy), 5, (255, 0, 0), -1)    
        
        #Areas of the contours
        area = cv2.contourArea(cnt)
        # contours_area.append(area*((mm_per_pixel)**2))
        contours_area.append(area)
        # print('Contours area = ',area*((mm_per_pixel)**2),' Center of the contour = ',center_x,center_y, ' w and h = ',w, h) #mm2 area
    
    # find the mm to pixel value using smallest contour
    mm_per_pixel=5/min(contour_width)
    
    #contour area in mm2
    contours_area = [x*((mm_per_pixel)**2) for x in contours_area]
   
    
    #diagonal in the picture and this is contant for the image
    min_distance_to_corner = math.sqrt((image_colour.shape[0])**2 + (image_colour.shape[0])**2) 
    nearest_corner=0 #nearest corner to contours
    
    #find the nearest corner to contours
    for corner in black_box:
        for i in range(len(cx_array)):
            distance = math.sqrt((cx_array[i] - corner[0])**2 + (cy_array[i] - corner[1])**2)
            if distance<min_distance_to_corner:
                min_distance_to_corner=distance
                nearest_corner=corner
    #declare the digit groups
    group1=[]
    group2=[]
    group3=[]
    #find the distances to contours from the found corner
    for i in range(len(cx_array)):
        distance = math.sqrt((cx_array[i] - nearest_corner[0])**2 + (cy_array[i] - nearest_corner[1])**2)
        distance=distance*mm_per_pixel
        # print('distance',distance)
        
        if distance <38:
            group1.append(i)
            # cv2.circle(image_colour, (cx_array[i],cy_array[i]), 5, (255, 0, 0), -1) 
        
        if 38 < distance < 63:
            group2.append(i)
            # cv2.circle(image_colour, (cx_array[i],cy_array[i]), 5, (0, 255, 0), -1)
        
        if 63 < distance:
            group3.append(i)
            # cv2.circle(image_colour, (cx_array[i],cy_array[i]), 5, (0, 0, 255), -1)    

    #calculate the credit
    def credit(group):
        if group:
            s=0
            m=0
            l=0
            for i in group:
                if contours_area[i] < 36:
                    s+=1
                if 36 < contours_area[i] < 84:
                    m+=1
                if 84 < contours_area[i]:
                    l+=1
            if s==1:
                digit=0
            if s==2:
                digit=1
            if s==3:
                digit=2
            if s==4:
                digit=3
            if m==1 and s==1:
                digit=4
            if m==1 and s==2:
                digit=5
            if m==1 and s==3:
                digit=6
            if l==1 and s==1:
                digit=7
            if l==1 and s==2:
                digit=8
            if l==1 and s==3:
                digit=9
        else:
            print(f"Please check the group[{i}]")
        return(digit) 
         
    
    finalcredit = int(f"{credit(group1)}{credit(group2)}{credit(group3)}")
    print(finalcredit)

    
    # Draw text on the image
    cv2.putText(
        image_colour,                    # image
        f"Colour-{colour}  Credit-{finalcredit}",                    # text
        (20, 40),               # bottom-left corner (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,  # font
        1.0,                    # font scale
        (0, 0, 0),            # color (B, G, R)
        2,                      # thickness
        cv2.LINE_AA             # line type
    )

        
    # Display images
    cv2.imshow("Image Details", image_colour)
    
    # Wait until 'Esc' key is pressed
    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
