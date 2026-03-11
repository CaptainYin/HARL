import numpy as np
from shapely.geometry import Polygon, Point,LineString,MultiLineString
import matplotlib.pyplot as plt
quad_segs=4



# Function to find endpoints and corners in the binary edge image
def find_endpoints(bin_image):
    # Kernel for morphological operations to identify endpoints
    # Using a cross kernel to consider connectivity from all sides
    cross_kernel = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]], dtype=np.uint8)

    # Dilate to enhance the connection points
    dilated = cv2.dilate(bin_image.astype(np.uint8), cross_kernel, iterations=1)

    # Now, isolate endpoints: dilation minus original will give connections
    endpoints = dilated - bin_image

    # Refine to get exact endpoints by and-ing with original
    endpoints = cv2.bitwise_and(bin_image, endpoints)

    # Detect corners
    corners = cv2.cornerHarris(bin_image.astype(np.float32), 2, 3, 0.04)
    corner_points = np.where(corners > 0.01 * corners.max())
    
    # Combine endpoints and corners
    yx = np.vstack(corner_points).T

    # Convert positions from arrays to list of tuples
    points = [tuple(point) for point in yx if bin_image[point[0], point[1]] == 1]
    
    return points

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Load the image from the provided file path
image_path = '/project/rl-collision-avoidance-master/worlds/testenv.png'
image = Image.open(image_path)

# Convert the image to grayscale
gray_image = image.convert('L')

# Convert the grayscale image to a numpy array
gray_array = np.array(gray_image)

# Binarize the image (find all black points)
# Assuming that the black color corresponds to low pixel values
threshold = 50  # Assuming black points are below this threshold
black_points = np.where(gray_array < threshold, 1, 0)

# # Plot the binarized image to visualize the black points
# plt.imshow(black_points, cmap='gray', interpolation='none')
# plt.title("Binarized Image Showing Black Points")
# plt.show()

# Find endpoints in the black points image
key_points = find_endpoints(black_points)

# Display the original image with the endpoints highlighted
fig, ax = plt.subplots()
ax.imshow(gray_array, cmap='gray', interpolation='none')
# print(key_points[0:7])
refined_keypoint=[]
for point in key_points:
    exist=0
    for po in refined_keypoint:
        if (point[0]-po[0])**2+(point[1]-po[1])**2<40:
            exist=1
        else:
            continue
    if exist==0:refined_keypoint.append(point)

print(len(refined_keypoint))
scenario_1=[]
for point in refined_keypoint:
    if point[1]<387:
        scenario_1.append(point)
scenario_2=[]
for point in refined_keypoint:
    if point[1]>387 and point[1]<677 and point[0]<394:
        scenario_2.append(point)

scenario_3=[]
for point in refined_keypoint:
    if point[1]>682 :
        scenario_3.append(point)

scenario_5=[]
for point in refined_keypoint:
    if point[0]>394 and point[0]<487:
        scenario_5.append(point)

scenario_6=[]
for point in refined_keypoint:
    if point[0]>446:
        scenario_6.append(point)


positions=[ 
##scenario_1_position
#     [-7.00,11.50,0.00,180.00],
# [-7.00,9.50,0.00,180.00],
# [-18.00,11.50,0.00,0.00],
# [-18.00,9.50,0.00,0.00],
# [-12.50,17.00,0.00,270.00],
# [-12.50,4.00,0.00,90.00],
##scenario_2_position
# [-2.00,16.00,0.00,-90.00],
# [0.00,16.00,0.00,-90.00],
# [3.00,16.00,0.00,-90.00],
# [5.00,16.00,0.00,-90.00],
##scenario_3_position
# [10.00,4.00,0.00,90.00],
# [12.00,4.00,0.00,90.00],
# [14.00,4.00,0.00,90.00],
# [16.00,4.00,0.00,90.00],
# [18.00,4.00,0.00,90.00],
##scenario_5_position
# [-2.5,-2.5,0.00,0.00],
# [-0.5,-2.5,0.00,0.00],
# [3.5,-2.5,0.00,180.00],
# [5.5,-2.5,0.00,180.00],
##scenario_6_position
# [-2.5,-18.5,0.00,90.00],
# [-0.5,-18.5,0.00,90.00],
# [1.5,-18.5,0.00,90.00],
# [3.5,-18.5,0.00,90.00],
# [5.5,-18.5,0.00,90.00],

##scenario_4_position
# [-6.00,-10.00,0.00,180.00],
# [-7.15,-6.47,0.00,216.00],
# [-10.15,-4.29,0.00,252.00],
# [-13.85,-4.29,0.00,288.00],
# [-16.85,-6.47,0.00,324.00],
# [-18.00,-10.00,0.00,360.00],
# [-16.85,-13.53,0.00,396.00],
# [-13.85,-15.71,0.00,432.00],
# [-10.15,-15.71,0.00,468.00],
# [-7.15,-13.53,0.00,504.00],

##scenario_7_position_bottom
# [10.00,-17.00,0.00,90.00],
# [12.00,-17.00,0.00,90.00],
# [14.00,-17.00,0.00,90.00],
# [16.00,-17.00,0.00,90.00],
# [18.00,-17.00,0.00,90.00],
##scenario_7_position_up
# [10.00,-2.00,0.00,-90.00],
# [12.00,-2.00,0.00,-90.00],
# [14.00,-2.00,0.00,-90.00],
# [16.00,-2.00,0.00,-90.00],
# [18.00,-2.00,0.00,-90.00]
]


ax.set_title("Key Points on Shapes")


scenario_2_lines=[[(138, 501), (138, 422), (58, 418),(58, 661),(137, 658),(138, 578),(298, 582),(298, 661),(377, 658),(378, 422),(298, 418), (297, 498),(138, 501)]]#  
scenario_1_lines=[[(18, 158), (158, 158), (158, 18), (238, 22), (238, 161), (378, 162), (377, 238), (238, 238), (237, 378), (158, 381), (158, 241), (18, 241),(18, 158)]]
scenario_3_lines=[[(78, 698),(378,702),(377, 978),(78, 978)],[(258,702),(258, 821)],[(257, 978),(258, 858)]]
scenario_5_lines=[[(418, 398), (418, 681), (477, 678), (478, 398),(418, 398)]]
scenario_1_position=[(-7.00,11.50),(-7.00,9.50),(-18.00,11.50),(-18.00,9.50),(-12.50,17.00),(-12.50,4.00)]
scenario_2_position=[(-2.00,16.00),(0.00,16.00),(3.00,16.00),(5.00,16.00)]
scenario_3_position=[(10.00,4.00),(12.00,4.00),(14.00,4.00),(16.00,4.00),(18.00,4.00)]
scenario_5_position=[(-2.5,-2.5),(-0.5,-2.5),(3.5,-2.5),(5.5,-2.5)]



ax.scatter([p[0]*25+500 for p in scenario_5_position], [-p[1]*20+400 for p in scenario_5_position], color='black', s=5)

# import math

# points = [(58, 418), (58, 661), (137, 658), (138, 422), (138, 501), (138, 578), (297, 498), (298, 418), (298, 582), (298, 661), (377, 658), (378, 422)]

# # Calculate the centroid
# centroid_x = sum(x for x, y in points) / len(points)
# centroid_y = sum(y for x, y in points) / len(points)

# # Function to calculate the angle from the centroid
# def angle_from_centroid(point):
#     x, y = point
#     return math.atan2(y - centroid_y, x - centroid_x)

# # Sort points based on the angle from the centroid
# sorted_points = sorted(points, key=angle_from_centroid)

# print(sorted_points)

# # scenario_1_position=[(-2.5,-18.5),(-0.5,-18.5),(1.5,-18.5),(3.5,-18.5),(5.5,-18.5)]

for coords in scenario_5_lines:
    for i in range(len(coords) - 1):
        ax.plot([coords[i][1],coords[i+1][1]], [coords[i][0],coords[i+1][0]], color='red', linewidth=1,linestyle="-") 




# ax.scatter([p[0] for p in scenario_6_position], [-p[1] for p in scenario_6_position], color='black', s=5)
# ax.scatter([p[0]*25+500 for p in positions], [-p[1]*20+400 for p in positions], color='black', s=5)
# sorted_data = sorted(data, key=lambda x: (x[0], x[1]))

# ax.scatter([p[1] for p in sorted_data], [p[0] for p in sorted_data], color='red', s=5)

# plt.plot(scenario_6[:,0],scenario_6[:,1])
plt.show()
# print(sorted_data)