import numpy as np
import cv2
from scipy.fftpack import dct

def encoder(video):

       # Encodes the video and returns a bitstream

    print("Encoding video...")

    block_size_luma = 16
    block_size_chroma = 8

    frames, height, width = video["Y"].shape
    num_blocks_h = height // block_size_luma
    num_blocks_w = width // block_size_luma


    blocks = {"Y": [], "U": [], "V": []}
    blocks ["Y"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_luma, block_size_luma), dtype=np.uint8)
    blocks ["U"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_chroma, block_size_chroma), dtype=np.uint8)
    blocks ["V"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_chroma, block_size_chroma), dtype=np.uint8)

    
    for f in range(frames):
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):

                block = video["Y"][f, i*block_size_luma:(i+1)*block_size_luma, j*block_size_luma:(j+1)*block_size_luma]
                blocks["Y"][f,i,j,:,:] = block

                block = video["U"][f, i*block_size_chroma:(i+1)*block_size_chroma, j*block_size_chroma:(j+1)*block_size_chroma]
                blocks["U"][f,i,j,:,:] = block


                block = video["V"][f, i*block_size_chroma:(i+1)*block_size_chroma, j*block_size_chroma:(j+1)*block_size_chroma]
                blocks["V"][f,i,j,:,:] = block
    

    blocks = intra_prediction(blocks)

    blocks[0] = apply_dct(blocks)   


    bitstream = blocks
    return bitstream


def vertical_intra_prediction(top):
    block_height, block_width = top.shape

    # Erstelle einen leeren Block für die Vorhersage
    prediction_block = np.zeros((block_height, block_width), dtype=np.uint8)

    # Iteriere über jede Spalte im Block
    for j in range(block_width):
        # Der Vorhersagewert wird aus dem oberen Nachbarpixel abgeleitet
        prediction_value = top[block_height-1, j]

        # Setze die Vorhersagewerte für die aktuelle Spalte im Block
        prediction_block[:, j] = prediction_value

    return prediction_block

def apply_vert_intraprediction(blocks):

    frames, num_blocks_h, num_blocks_w, block_size_luma, block_size_luma = blocks["Y"].shape
    block_size_chroma = blocks["U"].shape[3]
    temp_blocks = blocks

    residue_vert = {"Y": [], "U": [], "V": []}
    residue_vert ["Y"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_luma, block_size_luma), dtype=np.uint8)
    residue_vert ["U"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_chroma, block_size_chroma), dtype=np.uint8)
    residue_vert ["V"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_chroma, block_size_chroma), dtype=np.uint8)
     
    for f in range(frames):
        for i in range(num_blocks_h - 1):
            for j in range(num_blocks_w):

                blocks["Y"][f, i+1, j,:,:] = vertical_intra_prediction(temp_blocks["Y"][f, i, j,:,:])
                residue_vert["Y"][f, i+1, j,:,:] = temp_blocks["Y"][f, i+1, j,:,:] - blocks["Y"][f, i+1, j,:,:]

                blocks["U"][f, i+1, j,:,:] = vertical_intra_prediction(temp_blocks["U"][f, i, j,:,:])
                residue_vert["U"][f, i+1, j,:,:] = temp_blocks["U"][f, i+1, j,:,:] - blocks["U"][f, i+1, j,:,:]

                blocks["V"][f, i+1, j,:,:] = vertical_intra_prediction(temp_blocks["V"][f, i, j,:,:])
                residue_vert["V"][f, i+1, j,:,:] = temp_blocks["V"][f, i+1, j,:,:] - blocks["V"][f, i+1, j,:,:]
                
    return residue_vert


def horizontal_intra_prediction(left):
    
    block_height, block_width = left.shape

    # Erstelle einen leeren Block für die Vorhersage
    prediction_block = np.zeros((block_height, block_width), dtype=np.uint8)

    
    # Iteriere über jede Zeile im Block
    for i in range(block_height):
        # Der Vorhersagewert wird aus dem linken Nachbarpixel abgeleitet
        prediction_value = left[i, block_width-1]

        # Setze die Vorhersagewerte für die aktuelle Zeile im Block
        prediction_block[i, :] = prediction_value

    return prediction_block


def apply_hori_intraprediction(blocks):

    frames, num_blocks_h, num_blocks_w, block_size_luma, block_size_luma = blocks["Y"].shape
    block_size_chroma = blocks["U"].shape[3]
    temp_blocks = blocks

    residue_hori = {"Y": [], "U": [], "V": []}
    residue_hori ["Y"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_luma, block_size_luma), dtype=np.uint8)
    residue_hori ["U"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_chroma, block_size_chroma), dtype=np.uint8)
    residue_hori ["V"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_chroma, block_size_chroma), dtype=np.uint8)
     
    for f in range(frames):
        for i in range(num_blocks_h):
            for j in range(num_blocks_w - 1):

                blocks["Y"][f, i, j+1,:,:] = horizontal_intra_prediction(temp_blocks["Y"][f, i, j,:,:])
                residue_hori["Y"][f, i, j+1,:,:] = temp_blocks["Y"][f, i, j+1,:,:] - blocks["Y"][f, i, j+1,:,:]

                blocks["U"][f, i, j+1,:,:] = horizontal_intra_prediction(temp_blocks["U"][f, i, j,:,:])
                residue_hori["U"][f, i, j+1,:,:] = temp_blocks["U"][f, i, j+1,:,:] - blocks["U"][f, i, j+1,:,:]

                blocks["V"][f, i, j+1,:,:] = horizontal_intra_prediction(temp_blocks["V"][f, i, j,:,:])
                residue_hori["V"][f, i, j+1,:,:] = temp_blocks["V"][f, i, j+1,:,:] - blocks["V"][f, i, j+1,:,:]
                
    return residue_hori


def diagonal_intra_prediction(top, left, top_left):
    block_height, block_width = top.shape

    # Erstelle einen leeren Block für die Vorhersage
    prediction_block = np.zeros((block_height, block_width), dtype=np.uint8)
    
    for i in range(block_height):
        for j in range(block_width):

            if i == j:
                prediction_block[i, j] = top_left[block_height - 1, block_width - 1] 
            elif j > i:
                prediction_block[i, j] = top[block_height - 1, j - i - 1]
            elif i > j:
                prediction_block[i, j] = left[i - j - 1, block_width - 1] 

    return prediction_block


def apply_diag_intraprediction(blocks):

    frames, num_blocks_h, num_blocks_w, block_size_luma, block_size_luma = blocks["Y"].shape
    block_size_chroma = blocks["U"].shape[3]
    temp_blocks_1 = blocks
    temp_blocks_2 = blocks
    residue_diag = blocks

    '''
    residue_diag = {"Y": [], "U": [], "V": []}
    residue_diag ["Y"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_luma, block_size_luma), dtype=np.uint8)
    residue_diag ["U"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_chroma, block_size_chroma), dtype=np.uint8)
    residue_diag ["V"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_chroma, block_size_chroma), dtype=np.uint8)
    '''
     
    for f in range(frames):
        for i in range(num_blocks_h - 1):
            for j in range(num_blocks_w - 1):

                temp_blocks_2["Y"][f, i+1, j+1,:,:] = diagonal_intra_prediction(temp_blocks_1["Y"][f, i, j+1,:,:], temp_blocks_1["Y"][f, i+1, j,:,:], temp_blocks_1["Y"][f, i, j,:,:])
                residue_diag["Y"][f, i+1, j+1,:,:] = (blocks["Y"][f, i, j,:,:] - temp_blocks_2["Y"][f, i, j,:,:])

                temp_blocks_2["U"][f, i+1, j+1,:,:] = diagonal_intra_prediction(temp_blocks_1["U"][f, i, j+1,:,:], temp_blocks_1["U"][f, i+1, j,:,:], temp_blocks_1["U"][f, i, j,:,:])
                residue_diag["U"][f, i, j,:,:] = (blocks["U"][f, i, j,:,:] - temp_blocks_2["U"][f, i, j,:,:])

                temp_blocks_2["V"][f, i+1, j+1,:,:] = diagonal_intra_prediction(temp_blocks_1["V"][f, i, j+1,:,:], temp_blocks_1["V"][f, i+1, j,:,:], temp_blocks_1["V"][f, i, j,:,:])
                residue_diag["V"][f, i+1, j+1,:,:] = (blocks["V"][f, i, j,:,:] - temp_blocks_2["V"][f, i, j,:,:])

    return residue_diag



def intra_prediction(blocks):
    frames, num_blocks_h, num_blocks_w, block_size_luma, block_size_luma = blocks["Y"].shape
    block_size_chroma = blocks["U"].shape[3]

    temp_blocks_diag = blocks
    temp_blocks_vert = blocks
    temp_blocks_hori = blocks

    residue_diag = {"Y": [], "U": [], "V": []}
    residue_diag ["Y"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_luma, block_size_luma), dtype=np.uint8)
    residue_diag ["U"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_chroma, block_size_chroma), dtype=np.uint8)
    residue_diag ["V"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_chroma, block_size_chroma), dtype=np.uint8)

    residue_vert = {"Y": [], "U": [], "V": []}
    residue_vert ["Y"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_luma, block_size_luma), dtype=np.uint8)
    residue_vert ["U"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_chroma, block_size_chroma), dtype=np.uint8)
    residue_vert ["V"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_chroma, block_size_chroma), dtype=np.uint8)

    residue_hori = {"Y": [], "U": [], "V": []}
    residue_hori ["Y"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_luma, block_size_luma), dtype=np.uint8)
    residue_hori ["U"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_chroma, block_size_chroma), dtype=np.uint8)
    residue_hori ["V"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_chroma, block_size_chroma), dtype=np.uint8)

    residue = {"Y": [], "U": [], "V": []}
    residue ["Y"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_luma, block_size_luma), dtype=np.uint8)
    residue ["U"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_chroma, block_size_chroma), dtype=np.uint8)
    residue ["V"] = np.zeros((frames, num_blocks_h, num_blocks_w, block_size_chroma, block_size_chroma), dtype=np.uint8)

    prediction_mode = {"Y": [], "U": [], "V": []}
    prediction_mode ["Y"] = np.zeros((frames, num_blocks_h, num_blocks_w, 1), dtype=np.uint8)
    prediction_mode ["U"] = np.zeros((frames, num_blocks_h, num_blocks_w, 1), dtype=np.uint8)
    prediction_mode ["V"] = np.zeros((frames, num_blocks_h, num_blocks_w, 1), dtype=np.uint8)

    residue_diag = apply_diag_intraprediction(temp_blocks_diag)

    residue_vert = apply_vert_intraprediction(temp_blocks_vert)
    
    residue_hori = apply_hori_intraprediction(temp_blocks_hori)

    #berechne kleinsten prediction error, speichere den gewählten prediction mode mit kleinstem error in "prediction_mode"
    for f in range(frames):
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                
                if residue_diag["Y"][f,i,j,:,:] < residue_hori["Y"][f,i,j,:,:] and residue_diag["Y"][f,i,j,:,:] < residue_vert["Y"][f,i,j,:,:]:
                    residue["Y"][f,i,j,:,:] = residue_diag["Y"][f,i,j,:,:]
                    prediction_mode["Y"][f,i,j,:] = 1

                elif residue_vert["Y"][f,i,j,:,:] < residue_hori["Y"][f,i,j,:,:] and residue_vert["Y"][f,i,j,:,:] < residue_diag["Y"][f,i,j,:,:]:
                    residue["Y"][f,i,j,:,:] = residue_vert["Y"][f,i,j,:,:]
                    prediction_mode["Y"][f,i,j,:] = 2

                else:
                    residue["Y"][f,i,j,:,:] = residue_hori["Y"][f,i,j,:,:]
                    prediction_mode["Y"][f,i,j,:] = 3

                
                if residue_diag["U"][f,i,j,:,:] < residue_hori["U"][f,i,j,:,:] and residue_diag["U"][f,i,j,:,:] < residue_vert["U"][f,i,j,:,:]:
                    residue["U"][f,i,j,:,:] = residue_diag["U"][f,i,j,:,:]
                    prediction_mode["U"][f,i,j,:] = 1

                elif residue_vert["U"][f,i,j,:,:] < residue_hori["U"][f,i,j,:,:] and residue_vert["U"][f,i,j,:,:] < residue_diag["U"][f,i,j,:,:]:
                    residue["U"][f,i,j,:,:] = residue_vert["U"][f,i,j,:,:]
                    prediction_mode["U"][f,i,j,:] = 2
                    
                else:
                    residue["U"][f,i,j,:,:] = residue_hori["U"][f,i,j,:,:]
                    prediction_mode["U"][f,i,j,:] = 3


                if residue_diag["V"][f,i,j,:,:] < residue_hori["V"][f,i,j,:,:] and residue_diag["V"][f,i,j,:,:] < residue_vert["V"][f,i,j,:,:]:
                    residue["V"][f,i,j,:,:] = residue_diag["V"][f,i,j,:,:]
                    prediction_mode["V"][f,i,j,:] = 1

                elif residue_vert["V"][f,i,j,:,:] < residue_hori["V"][f,i,j,:,:] and residue_vert["V"][f,i,j,:,:] < residue_diag["V"][f,i,j,:,:]:
                    residue["V"][f,i,j,:,:] = residue_vert["V"][f,i,j,:,:]
                    prediction_mode["V"][f,i,j,:] = 2
                    
                else:
                    residue["V"][f,i,j,:,:] = residue_hori["V"][f,i,j,:,:]
                    prediction_mode["V"][f,i,j,:] = 3

    residues_with_prediction_mode = {residue, prediction_mode}

    return residues_with_prediction_mode


def dct_2d(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def apply_dct(blocks):
    '''
    nach intra_prediction wird ein neues array ausgegeben mit residues_with_prediction_mode = {residue, prediction_mode} und auf residue muss zugegriffen werden,
    also residues_with_prediction_mode[0] = blocks[0] = residue
    '''
    frames, num_blocks_h, num_blocks_w, a, b = blocks[0]["Y"].shape

    for f in range(frames):
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):

                blocks[0]["Y"][f, i, j,:,:] = dct_2d(blocks[0]["Y"][f, i, j,:,:])

                blocks[0]["U"][f, i, j,:,:] = dct_2d(blocks[0]["U"][f, i, j,:,:])
                
                blocks[0]["V"][f, i, j,:,:] = dct_2d(blocks[0]["V"][f, i, j,:,:])
    
    return blocks[0]



                

    









