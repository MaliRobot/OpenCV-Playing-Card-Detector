############## Playing Card Detector Functions ###############
#
# Author: Evan Juras
# Date: 9/5/17
# Description: Functions and classes for CardDetector.py that perform 
# various steps of the card detection algorithm


# Import necessary packages
import numpy as np
import cv2
import time

### Constants ###

# Adaptive threshold levels
BKG_THRESH = 60
CARD_THRESH = 30

# Width and height of card
CORNER_WIDTH = 600
CORNER_HEIGHT = 570

# Dimensions of rank train images
RANK_WIDTH = 400
RANK_HEIGHT = 580

RANK_DIFF_MAX = 40000

CARD_MAX_AREA = 300000
CARD_MIN_AREA = 60000

font = cv2.FONT_HERSHEY_SIMPLEX

### Structures to hold query card and train card information ###

### Structures to hold query card and train card information ###

class Query_card:
    """Structure to store information about query cards in the camera image."""

    def __init__(self):
        self.contour = [] # Contour of card
        self.width, self.height = 0, 0 # Width and height of card
        self.corner_pts = [] # Corner points of card
        self.center = [] # Center point of card
        self.warp = [] # 200x300, flattened, grayed, blurred image
        self.rank_img = [] # Thresholded, sized image of card's rank
        self.best_rank_match = "Unknown" # Best matched rank
        self.rank_diff = 0 # Difference between rank image and best matched train rank image

class Train_ranks:
    """Structure to store information about train rank images."""

    def __init__(self):
        self.img = [] # Thresholded, sized rank image loaded from hard drive
        self.name = "Placeholder"

### Functions ###
def load_ranks(filepath):
    """Loads rank images from directory specified by filepath. Stores
    them in a list of Train_ranks objects."""

    train_ranks = []
    i = 0
    
    for Rank in ['reito_lantern','ornate_kanzashi', 'free_from_the_real', 
              'sakura_tribe_scout', 'plains_ben_thomposon', 'path_of_angers_flame', 
              'sift_through_sands', 'setons_desire', 'phantom_nomad', 
              'divine_light', 'ghostly_wings', 'plains_fred_fields', 'locust_mister',
              'jugan_the_rising_star', 'whispering_shade', 'divergent_growth', 
              'ryusei_the_falling_star', 'dripping_tongue_zubera',
              'ninja_of_the _deep_hours', 'plains_matthew_mitchell', 'plains_greg_staples',
              'forest_quinton_hoover', 'forest_john_avon', 'ghost_lit_refeemer', 
              'kabuto_moth', 'kami_of_false_home', 'waxmane_baku', 'kami_of_tattered_shoji',
              'ethereal_haze', 'joyous_respite', 'orochi_sustainer', 'orochi_ranger',
              'commune_with_nature', 'petalmane_baku', 'scaled_hulk', 'harbinger_of_spring',
              'traproot_kami', 'rending_vines', 'vital_surge', 'torrent_of_stone',
              'descendant_of_soramaro', 'wandering_ones', 'orochi_sustainer', 'field_of_reality']:
        
        train_ranks.append(Train_ranks())
        train_ranks[i].name = Rank
        filename = Rank + '.jpg'
        train_ranks[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1

    return train_ranks

def preprocess_image(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    # The best threshold level depends on the ambient lighting conditions.
    # For bright lighting, a high threshold must be used to isolate the cards
    # from the background. For dim lighting, a low threshold must be used.
    # To make the card detector independent of lighting conditions, the
    # following adaptive threshold method is used.
    #
    # A background pixel in the center top of the image is sampled to determine
    # its intensity. The adaptive threshold is set at 50 (THRESH_ADDER) higher
    # than that. This allows the threshold to adapt to the lighting conditions.
    img_w, img_h = np.shape(image)[:2]
    bkg_level = gray[int(img_h/25)][int(img_w/2)]
    thresh_level = bkg_level + BKG_THRESH
    
    # try this for white border card
#    thresh_level = 100
    #

    retval, thresh = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)
    
    return thresh

def preprocess_white_image(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    
    img_w, img_h = np.shape(image)[:2]
    bkg_level = gray[int(img_h/25)][int(img_w/2)]
    thresh_level = bkg_level + BKG_THRESH

#    if thresh_level < 100:
    thresh_level = 110
    retval, thresh = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)
    
    return thresh

def find_cards(thresh_image, thresh_image_white):
    """Finds all card-sized contours in a thresholded camera image.
    Returns the number of cards, and a list of card contours sorted
    from largest to smallest."""
    # Find contours and sort their indices by contour size
    dummy,cnts,hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    dummy,cnts_wht,hier_wht = cv2.findContours(thresh_image_white,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)
    index_sort = [x for x in index_sort if cv2.contourArea(cnts[x]) > 50000]
    
    index_sort_wht = sorted(range(len(cnts_wht)), key=lambda i : cv2.contourArea(cnts_wht[i]),reverse=True)
    index_sort_wht = [x for x in index_sort_wht if cv2.contourArea(cnts_wht[x]) > 50000]

    # If there are no contours, do nothing
    if len(cnts) == 0 and len(cnts_wht) == 0:
        return [], []

    # decide if it's white or black border card
    if len(cnts) < len(cnts_wht):
        cnts = cnts_wht
        hier = hier_wht
        index_sort = index_sort_wht
    
    # Otherwise, initialize empty sorted contour and hierarchy lists
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts),dtype=int)

    # Fill empty lists with sorted contour and sorted hierarchy. Now,
    # the indices of the contour list still correspond with those of
    # the hierarchy list. The hierarchy array can be used to check if
    # the contours have parents or not.
    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    # Determine which of the contours are cards by applying the
    # following criteria: 1) Smaller area than the maximum card size,
    # 2), bigger area than the minimum card size, 3) have no parents,
    # and 4) have four corners
    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i],True)
        approx = cv2.approxPolyDP(cnts_sort[i],0.01*peri,True)

        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
            and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card

def get_treshold(Qcorner):
    white_level = Qcorner[15,int((CORNER_WIDTH)/2)]
    thresh_level = white_level - CARD_THRESH

    if (thresh_level <= 0):
        thresh_level = 1

    retval, query_thresh = cv2.threshold(Qcorner, thresh_level, 255, cv2.THRESH_TRUNC)
    return query_thresh

def preprocess_card(contour, image):
    """Uses contour to find information about the query card. Isolates images
    from the card."""

    # Initialize new Query_card object
    qCard = Query_card()

    qCard.contour = contour

    # Find perimeter of card and use it to approximate corner points
    peri = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.01*peri,True)
    pts = np.float32(approx)
    qCard.corner_pts = pts

    # Find width and height of card's bounding rectangle
    x,y,w,h = cv2.boundingRect(contour)
    qCard.width, qCard.height = w, h

    # Find center point of card by taking x and y average of the four corners.
    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCard.center = [cent_x, cent_y]

    # Warp card into 200x300 flattened image using perspective transform
    qCard.warp = flattener(image, pts, w, h)

    # Find rank contour and bounding rectangle, isolate and find largest contour
    dummy, qCard_cnts, hier = cv2.findContours(qCard.warp, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    qCard_cnts = sorted(qCard_cnts, key=cv2.contourArea,reverse=True)

    # Find bounding rectangle for largest contour, use it to resize query rank
    # image to match dimensions of the train rank image
    if len(qCard_cnts) != 0:
        x1,y1,w1,h1 = cv2.boundingRect(qCard_cnts[0])
        qCard_roi = qCard.warp[y1:y1+h1, x1:x1+w1]
        qCard_sized = cv2.resize(qCard_roi, (RANK_WIDTH,RANK_HEIGHT), 0, 0)
        qCard.rank_img = qCard_sized

    return qCard

def match_card(qCard, train_ranks):
    """Finds best rank matches for the query card. Differences
    the query card rank images with the train rank images.
    The best match is the rank image that has the least difference."""

    best_rank_match_diff = 40000
    best_rank_match_name = "Unknown"
    best_rank_name = None

    # If no contours were found in query card in preprocess_card function,
    # the img size is zero, so skip the differencing process
    # (card will be left as Unknown)
    if (len(qCard.rank_img) != 0):
        
        # Difference the query card rank image from each of the train rank images,
        # and store the result with the least difference
        for Trank in train_ranks:
            if Trank.img is None:
                continue
            
            diff_img = cv2.absdiff(cv2.equalizeHist(qCard.rank_img), cv2.equalizeHist(Trank.img))

            rank_diff = int(np.sum(diff_img)/255)
            print(rank_diff)
            if rank_diff < best_rank_match_diff:
                best_rank_match_diff = rank_diff
                best_rank_name = Trank.name

    # Combine best rank match and best suit match to get query card's identity.
    # If the best matches have too high of a difference value, card identity
    # is still Unknown
    if (best_rank_match_diff < RANK_DIFF_MAX):
        best_rank_match_name = best_rank_name

    # Return the identiy of the card and the quality of the suit and rank match
    return best_rank_match_name, best_rank_match_diff
    
    
def draw_results(image, qCard):
    """Draw the card name, center point, and contour on the camera image."""

    x = qCard.center[0]
    y = qCard.center[1]
    cv2.circle(image,(x,y),5,(255,0,0),-1)

    rank_name = qCard.best_rank_match

    # Draw card name twice, so letters have black outline
    cv2.putText(image,(rank_name),(x-60,y-10),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,(rank_name),(x-60,y-10),font,1,(50,200,200),2,cv2.LINE_AA)
    
    # Can draw difference value for troubleshooting purposes
    # (commented out during normal operation)
    #r_diff = str(qCard.rank_diff)
    #cv2.putText(image,r_diff,(x+20,y+30),font,0.5,(0,0,255),1,cv2.LINE_AA)

    return image

def flattener(image, pts, w, h):
    """Flattens an image of a card into a top-down 200x300 perspective.
    Returns the flattened, re-sized, grayed image.
    See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
    temp_rect = np.zeros((4,2), dtype = "float32")
    
    s = np.sum(pts, axis = 2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8*h: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.
    
    if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left
            
        
    maxWidth = 600
    maxHeight = 900

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

    return warp
