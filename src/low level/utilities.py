import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
from sklearn import preprocessing

# metadata files
file_bboxes = None
def initialize_bboxes_file(file_path):
    '''
        Initializes the bbox file so as to extract the information from it
    '''
    global file_bboxes
    file_bboxes = open(file_path, 'r')
    file_bboxes.readline()
    file_bboxes.readline()
def close_bboxes_file():
    '''
        Closes the bbox file
    '''
    file_bboxes.close()

def face_extraction(np_img):
    '''
        Takes the image and extracts the face using the bbox information
    '''
    [x, y, w, h] = map(int, file_bboxes.readline().replace('\n', '').split()[1:])
    return np_img[y:y+h+1, x:x+w+1]
def face_extraction_lfw(np_img):
    '''
        Didn't get the bbox for LFW dataset. So returning the image as is.
    '''
    return np_img

shape_predictor_name, len_shape_predictor, face_part_indices, dict_expand_percent = None, None, None, None
def initialize_dlib_shape_predictor(shape_predictor):
    '''
        Decide the pretrained dlib model to use
    '''
    global shape_predictor_name, len_shape_predictor, face_part_indices, dict_expand_percent
    shape_predictor_name = shape_predictor
    if shape_predictor.endswith("shape_predictor_5_face_landmarks.dat"):
        len_shape_predictor = 5
        face_part_indices = face_part_indices_5
    elif shape_predictor.endswith("shape_predictor_68_face_landmarks.dat"):
        len_shape_predictor = 68
        face_part_indices = face_part_indices_68
        dict_expand_percent = dict_expand_percent_68
    elif shape_predictor.endswith("shape_predictor_81_face_landmarks.dat"):
        len_shape_predictor = 81
        face_part_indices = face_part_indices_81
        dict_expand_percent = dict_expand_percent_81
    elif shape_predictor.endswith("shape_predictor_194_face_landmarks.dat"):
        len_shape_predictor = 194
        face_part_indices = face_part_indices_194
        dict_expand_percent = dict_expand_percent_194
    else: raise Exception("No such predictor present")

# For face alignment, please refer: https://github.com/HikkaV/Precise-face-alignment and https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
def face_alignment_dlib(np_img, display_intermediate_results = False, do_not_align = False):
    '''
        Aligns the input face using dlib
    '''
    # Detect face
    detector = dlib.get_frontal_face_detector()
    np_img_gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    rects = detector(np_img_gray, 0)
    if len(rects) == 0:
        raise Exception("DLIB has not detected faces")
    if len(rects) == 1:
        rect = rects[0]
    else:
        area, final_rect = 0, None
        for rect in rects:
            if (rect.right() - rect.left()) * (rect.bottom() - rect.top()) > area:
                area = (rect.right() - rect.left()) * (rect.bottom() - rect.top())
                final_rect = rect
        rect = final_rect

    # Detect face landmarks
    predictor = dlib.shape_predictor(shape_predictor_name)
    shape = predictor(np_img_gray, rect)
    shape = shape_to_np(shape)
    left_eye_center, right_eye_center = get_center_eyes_dlib(shape)
    center_of_forehead = np.vstack([left_eye_center, right_eye_center]).mean(axis = 0).round().astype(np.int)
    if display_intermediate_results:
        np_img_copy = np_img.copy()
        cv2.rectangle(np_img_copy, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 0, 255), 1)
        for i in range(len_shape_predictor):
            cv2.circle(np_img_copy, tuple(shape[i]), 2, (255, 0, 0), -1)
        cv2.line(np_img_copy, tuple(left_eye_center), tuple(right_eye_center), (0, 0, 255), 1)
        cv2.circle(np_img_copy, tuple(left_eye_center), 2, (0, 255, 0), -1)
        cv2.circle(np_img_copy, tuple(right_eye_center), 2, (0, 255, 0), -1)
        cv2.circle(np_img_copy, tuple(center_of_forehead), 2, (0, 255, 0), -1)
        plt.imshow(np_img_copy)
        plt.axis('off')
        plt.show()
    
    if do_not_align:
        return shape
    
    # find the angle of rotation
    dY, dX = (right_eye_center[1] - left_eye_center[1]) , (right_eye_center[0] - left_eye_center[0])
    angle = np.degrees(np.arctan2(dY, dX) / 2)
    
    # Apply the rotation
    M = cv2.getRotationMatrix2D(tuple(center_of_forehead), angle, 1) # rotation about center_of_forehead
    np_corners = np.array([[0, 0], [np_img.shape[1]-1, 0], [0, np_img.shape[0]-1], [np_img.shape[1]-1, np_img.shape[0]-1]]).astype(np.float)
    np_corners = (M @ np.append(np_corners, np.ones(shape = (len(np_corners), 1)), axis = 1).T).T
    [x_min, y_min], [x_max, y_max] = np_corners.min(axis = 0), np_corners.max(axis = 0)
    M[0, 2] = M[0, 2] - x_min; M[1, 2] = M[1, 2] - y_min # translation
    rotated_image = cv2.warpAffine(np_img, M, (int(np.round(x_max - x_min)), int(np.round(y_max - y_min))), flags = cv2.INTER_CUBIC)

    # rotating shape points
    shape = (M @ np.append(shape, np.ones(shape = (len_shape_predictor, 1)), axis = 1).T).T

    return rotated_image, shape.round().astype(np.int)

shape_to_np = lambda shape: np.array([[shape.part(i).x, shape.part(i).y] for i in range(len_shape_predictor)]).round().astype(np.int) # Converts dlib model to numpy

get_center_eyes_dlib = lambda shape: (np.mean(shape[face_part_indices("left_eye")], axis = 0).round().astype(int), np.mean(shape[face_part_indices("right_eye")], axis = 0).round().astype(int)) # fetch the center coordinates of the left eye and right eye

def rectangular_face_part_extraction(np_img, locations, do_expand_face_region, face_part):
    '''
        Extracts a rectangular face part
    '''
    [x_min, y_min], [x_max, y_max] = locations.min(axis = 0), locations.max(axis = 0)
    if do_expand_face_region:
        [x_min, y_min, x_max, y_max] = expand_face_region([x_min, y_min, x_max, y_max], dict_expand_percent, face_part)
    return np_img[y_min:y_max+1, x_min:x_max+1].copy()
def polygon_face_part_extraction(np_img, locations):
    '''
        Extracts a polygon face part
    '''
    mask = np.zeros(np_img.shape[:2])
    cv2.fillConvexPoly(mask, locations, 1)
    mask = mask.astype(np.bool)
    out = np.zeros_like(np_img)
    out[mask] = np_img[mask]
    return out

# Returns the indices of the face parts
def face_part_indices_5(face_part):
    if face_part == "left_eye":
        return [2, 3]
    elif face_part == "right_eye":
        return [0, 1]
    elif face_part == "nose":
        return [4]
    else: raise Exception("Error: Invalid face_part for the chosen predictor")
def face_part_indices_68(face_part):
    if face_part in ["forehead", "hair"]: raise Exception("Error: Invalid face_part for the chosen predictor")
    return face_part_indices_81(face_part)
def face_part_indices_81(face_part):
    indices = None
    if face_part == "eyes":
        indices = face_part_indices_81("left_eye")
        indices.extend(face_part_indices_81("right_eye"))
    elif face_part == "left_eye":
        indices = list(range(36, 42))
    elif face_part == "right_eye":
        indices = list(range(42, 48))
    elif face_part == "nose":
        indices = list(range(27, 36))
    elif face_part == "mouth":
        indices = list(range(48, 65))
    elif face_part == "eyes_eyebrows":
        indices = face_part_indices_81("eyes")
        indices.extend(list(range(17, 27)))
    elif face_part == "left_eye_eyebrow":
        indices = face_part_indices_81("left_eye")
        indices.extend(list(range(17, 22)))
    elif face_part == "right_eye_eyebrow":
        indices = face_part_indices_81("right_eye")
        indices.extend(list(range(22, 27)))
    elif face_part == "forehead":
        indices = [19, 24, 72, 69]
    elif face_part == "chin":
        indices = [6, 8, 11, 57]
    elif face_part == "moustache":
        indices = [50, 52, 33]
    elif face_part == "left_cheek":
        indices = [1, 6, 48]
    elif face_part == "right_cheek":
        indices = [16, 11, 54]
    elif face_part == "hair":
        indices = [73, 76]
    elif face_part == "full_face":
        indices = list(range(len_shape_predictor))
    else: raise Exception("Error: Invalid face_part for the chosen predictor")
    return indices
def face_part_indices_194(face_part):
    if face_part == "nose":
        result = list(range(135, 152))
    elif face_part == "mouth":
        result = list(range(11, 21))
        result.extend(list(range(22, 26)))
        result.extend(list(range(152, 194)))
    elif face_part == "eyes":
        result = face_part_indices_194("left_eye")
        result.extend(face_part_indices_194("right_eye"))
    elif face_part == "eyebrows":
        result = face_part_indices_194("left_eyebrow")
        result.extend(face_part_indices_194("right_eyebrow"))
    elif face_part == "eyes_eyebrows":
        result = face_part_indices_194("eyes")
        result.extend(face_part_indices_194("eyebrows"))
    elif face_part == "left_eye":
        result = list(range(48, 54))
        result.extend(list(range(55, 65)))
        result.extend(list(range(66, 70)))
    elif face_part == "right_eye":
        result = list(range(26, 32))
        result.extend(list(range(33, 43)))
        result.extend(list(range(44, 48)))
    elif face_part == "left_eyebrow":
        result = list(range(92, 98))
        result.extend(list(range(99, 109)))
        result.extend(list(range(110, 114)))
    elif face_part == "right_eyebrow":
        result = list(range(70, 76))
        result.extend(list(range(77, 87)))
        result.extend(list(range(88, 92)))
    elif face_part == "left_eye_eyebrow":
        result = face_part_indices_194("left_eye")
        result.extend(face_part_indices_194("left_eyebrow"))
    elif face_part == "right_eye_eyebrow":
        result = face_part_indices_194("right_eye")
        result.extend(face_part_indices_194("right_eyebrow"))
    elif face_part == "chin":
        result = [32, 76, 87, 98, 109, 114, 118, 173, 172]
    elif face_part == "moustache":
        result = [142, 143, 144, 156, 157, 158]
    elif face_part == "left_cheek":
        result = [0, 32, 152]
    elif face_part == "right_cheek":
        result = [118, 134, 164]
    elif face_part == "full_face":
        result = list(range(len_shape_predictor))
    else: raise Exception("Error: Invalid face_part for the chosen predictor")
    return result
def face_part_extraction(np_img, face_part, shape, extracted_shape = "rect", do_expand_face_region = True):
    '''
        Extracts the face part
    '''
    indices = face_part_indices(face_part)
    locations = shape[indices]
    if extracted_shape == "rect":
        if face_part == "forehead": # for forehead, we don't want the eyebrows to show. Hence, manipulating the data
            if locations[0][1] < locations[1][1]: locations = locations[[0, 2, 3], :]
            else: locations = locations[[1, 2, 3], :]
        return rectangular_face_part_extraction(np_img, locations, do_expand_face_region, face_part)
    elif extracted_shape == "poly":
        return polygon_face_part_extraction(np_img, locations)
    else:
        raise Exception("Error: Invalid shape of extracted image")

dict_expand_percent_68 = {"eyes_eyebrows": {"left": 0, "right": 0, "top": 0.01, "bottom": 0.1}, 
                        "nose": {"left": 0.1, "right": 0.1, "top": 0, "bottom": 0}, 
                        "eyes": {"left": 0.05, "right": 0.05, "top": 0.01, "bottom": 0.1}, 
                        "mouth": {"left": 0.05, "right": 0.05, "top": 0.01, "bottom": 0.01}, 
                        "left_eye_eyebrow": {"left": 0, "right": 0, "top": 0.01, "bottom": 0.1}, 
                        "right_eye_eyebrow": {"left": 0, "right": 0, "top": 0.01, "bottom": 0.1},
                        "chin": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                        "moustache": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                        "left_cheek": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                        "right_cheek": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                        "full_face": {"left": 0, "right": 0, "top": 0, "bottom": 0}
                    }
dict_expand_percent_81 = {"eyes_eyebrows": {"left": 0, "right": 0, "top": 0.01, "bottom": 0.1}, 
                        "nose": {"left": 0.1, "right": 0.1, "top": 0, "bottom": 0}, 
                        "eyes": {"left": 0.05, "right": 0.05, "top": 0.01, "bottom": 0.1}, 
                        "mouth": {"left": 0.05, "right": 0.05, "top": 0.01, "bottom": 0.01},
                        "left_eye_eyebrow": {"left": 0, "right": 0, "top": 0.01, "bottom": 0.1}, 
                        "right_eye_eyebrow": {"left": 0, "right": 0, "top": 0.01, "bottom": 0.1},
                        "chin": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                        "moustache": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                        "left_cheek": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                        "right_cheek": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                        "forehead": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                        "full_face": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                        "hair": {"left": 0.1, "right": 0.1, "top": 0.3, "bottom": 0}
                    }
dict_expand_percent_194 = {"eyes_eyebrows": {"left": 0, "right": 0, "top": 0.01, "bottom": 0.1}, 
                        "nose": {"left": 0.02, "right": 0.02, "top": 0.08, "bottom": 0}, 
                        "eyes": {"left": 0.05, "right": 0.05, "top": 0.01, "bottom": 0.1}, 
                        "mouth": {"left": 0.05, "right": 0.05, "top": 0.01, "bottom": 0.01}, 
                        "eyebrows": {"left": 0, "right": 0, "top": 0.01, "bottom": 0.01},
                        "left_eye_eyebrow": {"left": 0, "right": 0, "top": 0.01, "bottom": 0.1}, 
                        "right_eye_eyebrow": {"left": 0, "right": 0, "top": 0.01, "bottom": 0.1},
                        "chin": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                        "moustache": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                        "left_cheek": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                        "right_cheek": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                        "full_face": {"left": 0, "right": 0, "top": 0, "bottom": 0}
                    }
def expand_face_region(bbox, dict_percent, face_part):
    '''
        Expands the face region by the mentioned percentage
    '''
    #bbox = x1, y1, x2, y2 (top left and bottom right)
    x1 = (1 - dict_percent[face_part]["left"]) * bbox[0]
    y1 = (1 - dict_percent[face_part]["top"]) * bbox[1]
    x2 = (1 + dict_percent[face_part]["right"]) * bbox[2]
    y2 = (1 + dict_percent[face_part]["bottom"]) * bbox[3]
    return int(np.round(x1)), int(np.round(y1)), int(np.round(x2)), int(np.round(y2))

def normalize(np_img, normalization_type):
    '''
        Normalize the input data (image data)
    '''
    if normalization_type == "robust":
        preprocessor = preprocessing.RobustScaler()
    elif normalization_type == "standard":
        preprocessor = preprocessing.StandardScaler()
    elif normalization_type == "minmax":
        preprocessor = preprocessing.MinMaxScaler()
    else:
        raise Exception("Error: Invalid normalization type")
    if len(np_img.shape) == 3 and np_img.shape[2] == 3:
        return preprocessor.fit_transform(np_img.reshape(-1, 3)).reshape(np_img.shape)
    elif len(np_img.shape) == 2:
        return preprocessor.fit_transform(np_img.reshape(-1, 1)).reshape(np_img.shape)
    else:
        raise Exception("Error: Invalid image shape")

HISTOGRAM_BINS = 50
def extract_low_level_features_research_paper(img_face_aligned, shape, display_intermediate_results = False):
    '''
        Extracts the low level feature
    '''
    # Face part extraction (Eyes)
    img_left_eye = face_part_extraction(img_face_aligned, "left_eye_eyebrow", shape, extracted_shape = "rect")
    img_right_eye = face_part_extraction(img_face_aligned, "right_eye_eyebrow", shape, extracted_shape = "rect")
    # img_eyes = np.concatenate((img_left_eye, img_right_eye), axis = 1)
    # or
    # img_eyes = cv2.resize(face_part_extraction(img_face_aligned, "eyes_eyebrows", shape, extracted_shape = "rect"), (80, 30))

    # Face part extraction (Nose)
    img_nose = face_part_extraction(img_face_aligned, "nose", shape, extracted_shape = "rect")
    
    # Face part extraction (Mouth)
    img_mouth = face_part_extraction(img_face_aligned, "mouth", shape, extracted_shape = "rect")
    
    # Face part extraction (Chin)
    img_chin = face_part_extraction(img_face_aligned, "chin", shape, extracted_shape = "rect")
    
    # Face part extraction (Moustache)
    img_moustache = face_part_extraction(img_face_aligned, "moustache", shape, extracted_shape = "rect")
    
    # Face part extraction (Cheeks)
    img_left_cheek = face_part_extraction(img_face_aligned, "left_cheek", shape, extracted_shape = "rect")
    img_right_cheek = face_part_extraction(img_face_aligned, "right_cheek", shape, extracted_shape = "rect")

    # Face part extraction (Full face)
    img_full_face = face_part_extraction(img_face_aligned, "full_face", shape, extracted_shape = "rect")
    
    # Face part extraction (Forehead)
    if len_shape_predictor == 81:
        img_forehead = face_part_extraction(img_face_aligned, "forehead", shape, extracted_shape = "rect")
        img_hair = face_part_extraction(img_face_aligned, "hair", shape, extracted_shape = "rect")

    if display_intermediate_results:
        if len_shape_predictor == 81:
            fig, axs = plt.subplots(1, 12, figsize=(30,5))
        else:
            fig, axs = plt.subplots(1, 10, figsize=(30,5))
        axs[0].imshow(img_face_aligned)
        axs[1].imshow(img_left_eye)
        axs[2].imshow(img_right_eye)
        axs[3].imshow(img_nose)
        axs[4].imshow(img_mouth)
        axs[5].imshow(img_chin)
        axs[6].imshow(img_moustache)
        axs[7].imshow(img_left_cheek)
        axs[8].imshow(img_right_cheek)
        axs[9].imshow(img_full_face)
        if len_shape_predictor == 81: axs[10].imshow(img_forehead)
        if len_shape_predictor == 81: axs[11].imshow(img_hair)
        axs[0].set_title('Aligned Face')
        axs[1].set_title('Left eye')
        axs[2].set_title('Right eye')
        axs[3].set_title('Nose')
        axs[4].set_title('Mouth')
        axs[5].set_title('Chin')
        axs[6].set_title('Moustache')
        axs[7].set_title('Left cheek')
        axs[8].set_title('Right cheek')
        axs[9].set_title('Full face')
        if len_shape_predictor == 81: axs[10].set_title('Forehead')
        if len_shape_predictor == 81: axs[11].set_title('Hair')
        axs[0].axis('off')
        axs[1].axis('off')
        axs[2].axis('off')
        axs[3].axis('off')
        axs[4].axis('off')
        axs[5].axis('off')
        axs[6].axis('off')
        axs[7].axis('off')
        axs[8].axis('off')
        axs[9].axis('off')
        if len_shape_predictor == 81: axs[10].axis('off')
        if len_shape_predictor == 81: axs[11].axis('off')
        plt.show()

    def extract_low_level_features(np_img, display_intermediate_results = False):
        rgb_hist_feature = np.histogram(normalize(np_img, "standard").flatten(), bins = HISTOGRAM_BINS)[0]
        hsv_hist_feature = np.histogram(normalize(cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV), "standard"), bins = HISTOGRAM_BINS)[0]
        sobel_x = cv2.Sobel(cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY), cv2.CV_64F, 1, 0, ksize=7)
        sobel_y = cv2.Sobel(cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY), cv2.CV_64F, 0, 1, ksize=7)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        grad_dir = np.arctan2(sobel_y, sobel_x)
        if display_intermediate_results:
            fig, axs = plt.subplots(1, 2, figsize=(12, 3))
            axs[0].imshow(grad_mag, cmap = 'gray')
            axs[1].imshow(grad_dir, cmap = 'gray')
            axs[0].axis('off')
            axs[1].axis('off')
            plt.show()
        grad_mag_hist_feature = np.histogram(normalize(grad_mag, "standard").flatten(), bins = HISTOGRAM_BINS)[0]
        grad_dir_hist_feature = np.histogram(normalize(grad_dir, "standard").flatten(), bins = HISTOGRAM_BINS)[0]
        return rgb_hist_feature, hsv_hist_feature, grad_mag_hist_feature, grad_dir_hist_feature

    # display_intermediate_results = False
    
    # Eyes
    rgb_left_eye, hsv_left_eye, grad_mag_left_eye, grad_orien_left_eye = extract_low_level_features(img_left_eye, display_intermediate_results)
    rgb_right_eye, hsv_right_eye, grad_mag_right_eye, grad_orien_right_eye = extract_low_level_features(img_right_eye, display_intermediate_results)

    # Nose
    rgb_nose, hsv_nose, grad_mag_nose, grad_orien_nose = extract_low_level_features(img_nose, display_intermediate_results)

    # Mouth
    rgb_mouth, hsv_mouth, grad_mag_mouth, grad_orien_mouth = extract_low_level_features(img_mouth, display_intermediate_results)
    
    # Chin
    rgb_chin, hsv_chin, grad_mag_chin, grad_orien_chin = extract_low_level_features(img_chin, display_intermediate_results)
    
    # Moustache
    rgb_moustache, hsv_moustache, grad_mag_moustache, grad_orien_moustache = extract_low_level_features(img_moustache, display_intermediate_results)
    
    # Cheeks
    rgb_left_cheek, hsv_left_cheek, grad_mag_left_cheek, grad_orien_left_cheek = extract_low_level_features(img_left_cheek, display_intermediate_results)
    rgb_right_cheek, hsv_right_cheek, grad_mag_right_cheek, grad_orien_right_cheek = extract_low_level_features(img_right_cheek, display_intermediate_results)
    
    # Forehead
    rgb_forehead, hsv_forehead, grad_mag_forehead, grad_orien_forehead = extract_low_level_features(img_forehead, display_intermediate_results)

    # Full face
    rgb_full_face, hsv_full_face, grad_mag_full_face, grad_orien_full_face = extract_low_level_features(img_full_face, display_intermediate_results)

    # Hair
    rgb_hair, hsv_hair, grad_mag_hair, grad_orien_hair = extract_low_level_features(img_hair, display_intermediate_results)

    return rgb_left_eye, hsv_left_eye, grad_mag_left_eye, grad_orien_left_eye, \
        rgb_right_eye, hsv_right_eye, grad_mag_right_eye, grad_orien_right_eye, \
        rgb_nose, hsv_nose, grad_mag_nose, grad_orien_nose, \
        rgb_mouth, hsv_mouth, grad_mag_mouth, grad_orien_mouth, \
        rgb_chin, hsv_chin, grad_mag_chin, grad_orien_chin, \
        rgb_moustache, hsv_moustache, grad_mag_moustache, grad_orien_moustache, \
        rgb_left_cheek, hsv_left_cheek, grad_mag_left_cheek, grad_orien_left_cheek, \
        rgb_right_cheek, hsv_right_cheek, grad_mag_right_cheek, grad_orien_right_cheek, \
        rgb_forehead, hsv_forehead, grad_mag_forehead, grad_orien_forehead, \
        rgb_full_face, hsv_full_face, grad_mag_full_face, grad_orien_full_face, \
        rgb_hair, hsv_hair, grad_mag_hair, grad_orien_hair