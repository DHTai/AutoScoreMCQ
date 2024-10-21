import cv2
import argparse
import imutils
import numpy as np

def read_image(img):
    return cv2.imread(img)

def convert_image_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def threshold_image(gray):
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def invert_image(threadholed):
    return  cv2.bitwise_not(threadholed)

def dilate_image(inverted):
    return cv2.dilate(inverted, None, iterations=5)

def print_image(name, img):
    cv2.imwrite(name + '.jpg', img)

def find_contours(src, dilated):
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_with_all_contours = src.copy()
    cv2.drawContours(image_with_all_contours, contours, -1, (0, 255, 0), 3)
    return image_with_all_contours, contours

def filter_contours_and_leave_only_rectangles(src, contours):
    rectangular_contours = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            rectangular_contours.append(approx)
    image_with_only_rectangular_contours = src.copy()
    cv2.drawContours(image_with_only_rectangular_contours, rectangular_contours, -1, (0, 255, 0), 3)
    return rectangular_contours, image_with_only_rectangular_contours

def find_largest_contour_by_area(src, rectangular_contours):
    max_area = 0
    contour_with_max_area = None
    for contour in rectangular_contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            contour_with_max_area = contour
    image_with_contour_with_max_area = src.copy()
    cv2.drawContours(image_with_contour_with_max_area, [contour_with_max_area], -1, (0, 255, 0), 3)
    return contour_with_max_area, image_with_contour_with_max_area

def sort_contours(cnts):
	return sorted(cnts, key=cv2.contourArea, reverse=True)

def get_large_contours(src, rectangular_contours):
    sorted_cnts = sort_contours(rectangular_contours)
    image_with_contour_with_large_area = src.copy()
    cv2.drawContours(image_with_contour_with_large_area, [sorted_cnts[0], sorted_cnts[1], sorted_cnts[2]], -1, (0, 255, 0), 3)
    return image_with_contour_with_large_area, [sorted_cnts[0], sorted_cnts[1], sorted_cnts[2]]

def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def order_points_in_the_contour(src, cnt):
    contour_order_points = order_points(cnt)
    image_with_points_plotted = src.copy()
    for point in contour_order_points:
        point_coordinates = (int(point[0]), int(point[1]))
        image_with_points_plotted = cv2.circle(image_with_points_plotted, point_coordinates, 10, (0, 0, 255), -1)
    return image_with_points_plotted, contour_order_points

def calculateDistanceBetween2Points(p1, p2):
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

def calculate_new_width_and_height_of_image(src, points):
    existing_image_width = src.shape[1]
    existing_image_width_reduced_by_10_percent = int(existing_image_width * 0.9)
    
    distance_between_top_left_and_top_right = calculateDistanceBetween2Points(points[0], points[1])
    distance_between_top_left_and_bottom_left = calculateDistanceBetween2Points(points[0], points[3])
    aspect_ratio = distance_between_top_left_and_bottom_left / distance_between_top_left_and_top_right
    new_image_width = existing_image_width_reduced_by_10_percent
    new_image_height = int(new_image_width * aspect_ratio)
    return new_image_width, new_image_height

def apply_perspective_transform(src, cnt, points, w, h):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    perspective_corrected_image = cv2.warpPerspective(src, matrix, (w, h))
    return perspective_corrected_image

def add_10_percent_padding(src, table):
    h = src.shape[0]
    padding = int(h * 0.1)
    perspective_corrected_image_with_padding = cv2.copyMakeBorder(table, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return perspective_corrected_image_with_padding

def erode_vertical_lines(inverted_image):
    hor = np.array([[1,1,1,1,1,1]])
    vertical_lines_eroded_image = cv2.erode(inverted_image, hor, iterations=10)
    vertical_lines_eroded_image = cv2.dilate(vertical_lines_eroded_image, hor, iterations=10)
    return vertical_lines_eroded_image

def erode_horizontal_lines(inverted_image):
    ver = np.array([[1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1]])
    horizontal_lines_eroded_image = cv2.erode(inverted_image, ver, iterations=10)
    horizontal_lines_eroded_image = cv2.dilate(horizontal_lines_eroded_image, ver, iterations=10)
    return horizontal_lines_eroded_image

def combine_eroded_images(vertical_lines, horizontal_lines):
    return cv2.add(vertical_lines, horizontal_lines)

def dilate_combined_image_to_make_lines_thicker(combine_line_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    combined_image_dilated = cv2.dilate(combine_line_image, kernel, iterations=5)
    return combined_image_dilated

def subtract_combined_and_dilated_image_from_original_image(inverted_image, combined_image_dilated):
    image_without_lines = cv2.subtract(inverted_image, combined_image_dilated)
    return image_without_lines

def remove_noise_with_erode_and_dilate(image_without_lines):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image_without_lines_noise_removed = cv2.erode(image_without_lines, kernel, iterations=1)
    image_without_lines_noise_removed = cv2.dilate(image_without_lines_noise_removed, kernel, iterations=1)
    return image_without_lines_noise_removed

def threshold_table(grey):
    return cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)[1]

def dilate_table(thresholded_image):
    kernel_to_remove_gaps_between_words = np.array([
            [1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1]
            # [1,1,1,1,1,1,1,1,1],
            # [1,1,1,1,1,1,1,1,1]
     ])
    #kernel_to_remove_gaps_between_words = np.ones([10, 2])
    dilated_image = cv2.dilate(thresholded_image, kernel_to_remove_gaps_between_words, iterations=5)
    simple_kernel = np.ones((5,5), np.uint8)
    dilated_image = cv2.dilate(dilated_image, simple_kernel, iterations=2)
    return dilated_image

def find_table_contours(original_image, dilated_image):
    result = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = result[0]
    image_with_contours_drawn = original_image.copy()
    cv2.drawContours(image_with_contours_drawn, contours, -1, (0, 255, 0), 3)
    return image_with_contours_drawn, contours

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--image", required=True, help="the image path")
    argparser.add_argument("--gray", action='store_true', help="print grayscale image")
    argparser.add_argument("--threshold", action='store_true', help="print threshold image")
    argparser.add_argument("--inverted", action='store_true', help="print inverted image")
    argparser.add_argument("--dilated", action='store_true', help="print dilated image")
    argparser.add_argument("--contours", action='store_true', help="print image with all countours image")
    argparser.add_argument("--OnlyRectangle", action='store_true', help="print filter contours and leave only rectangle")
    argparser.add_argument("--LargestContour", action='store_true', help="print largest contour by area")
    argparser.add_argument("--PrintLargeContour", action="store_true", help="print large contours")
    argparser.add_argument("--PrintPointLargeContour", action="store_true", help="print point of large contours")
    args = vars(argparser.parse_args())

    img = read_image(args["image"])
    MAX_TABLE = 3

    gray_img = convert_image_to_grayscale(img)
    if args["gray"]:
        print_image("gray image", gray_img)
    
    threshold_img = threshold_image(gray_img)
    if args["threshold"]:
        print_image("threshold image", threshold_img)

    inverted_img = invert_image(threshold_img)
    if args["inverted"]:
        print_image("inverted image", inverted_img)
    
    dilated_img = dilate_image(inverted_img)
    if args["dilated"]:
        print_image("dilated image", dilated_img)
    
    all_contour_img, cnts = find_contours(img, dilated_img)
    if args["contours"]:
        print_image("all contours image", all_contour_img)

    rect_cnts, only_rect_cnts_img = filter_contours_and_leave_only_rectangles(img, cnts)
    if args["OnlyRectangle"]:
        print_image("only rectangular contours image", only_rect_cnts_img)

    contour_with_max_area, contour_with_max_area_img = find_largest_contour_by_area(img, rect_cnts)
    if args["LargestContour"]:
        print_image("largest contour", contour_with_max_area_img)
    
    large_contours_img, cnt_list = get_large_contours(img, rect_cnts)
    if args["PrintLargeContour"]:
        print_image("large contours image", large_contours_img)

    old_img = img
    table_list = []
    for cnt in cnt_list:
        plotting_order_img, points = order_points_in_the_contour(old_img, cnt)
        old_img = plotting_order_img
        w,h = calculate_new_width_and_height_of_image(img, points)
        new_img = apply_perspective_transform(img, cnt, points, w, h)
        table_list.append(new_img)

    if args["PrintPointLargeContour"]:
        print_image("point of large contours image", plotting_order_img)

    count = 1
    for table in table_list:
        table = add_10_percent_padding(img, table)
        gray_table = convert_image_to_grayscale(table)
        thresholded_table = threshold_table(gray_table)
        inverted_table = invert_image(thresholded_table)
        vertical_lines_eroded_table = erode_vertical_lines(inverted_table)
        horizontal_lines_eroded_table = erode_horizontal_lines(inverted_table)
        combine_lines_eroded_table = combine_eroded_images(vertical_lines_eroded_table, horizontal_lines_eroded_table)
        combined_table_dilated = dilate_combined_image_to_make_lines_thicker(combine_lines_eroded_table)
        table_without_lines = subtract_combined_and_dilated_image_from_original_image(inverted_table, combined_table_dilated)
        table_without_lines_noise_removed = remove_noise_with_erode_and_dilate(table_without_lines)
        dilated_table = dilate_table(thresholded_table)
        table_with_contours_drawn, table_cnts = find_table_contours(table, dilated_table)
        table = table_without_lines_noise_removed
        print_image("Table" + str(count), table)
        count = count + 1
    