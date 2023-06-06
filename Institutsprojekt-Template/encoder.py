import numpy as np
import cv2
from scipy.fftpack import dct

def encoder(video):

       # Encodes the video and returns a bitstream

    print("Encoding video...")

    block_size_luma = 16
    block_size_chroma = 8

    frames, height, width = video["Y"].shape # bsp : video["Y"][64, 384, 384]
    num_blocks_h_luma = height // block_size_luma
    num_blocks_w_luma = width // block_size_luma
    num_blocks_perframe_luma = num_blocks_h_luma * num_blocks_w_luma

    frames, height, width = video["U"].shape
    num_blocks_h_chroma = height // block_size_chroma
    num_blocks_w_chroma = width // block_size_chroma
    num_blocks_perframe_chroma = num_blocks_h_chroma * num_blocks_w_chroma


    blocks = {"Y": [], "U": [], "V": []}

    blocks ["Y"] = np.zeros((frames, num_blocks_perframe_luma, block_size_luma, block_size_luma), dtype=np.uint8)
    blocks ["U"] = np.zeros((frames, num_blocks_perframe_chroma, block_size_chroma, block_size_chroma), dtype=np.uint8)
    blocks ["V"] = np.zeros((frames, num_blocks_perframe_chroma, block_size_chroma, block_size_chroma), dtype=np.uint8)

    #Y-Blöcke
    for f in range(frames):
        numblock = 0
        for i in range(num_blocks_h_luma):
            for j in range(num_blocks_w_luma):
                block = video["Y"][f, i*block_size_luma:(i+1)*block_size_luma, j*block_size_luma:(j+1)*block_size_luma]
             
                block = intra_prediction(block)
                #block = dct(block)
                blocks["Y"][f,numblock,:,:] = block
                numblock += 1
     
    #U-Blöcke
    for f in range(frames):
        numblock = 0
        for i in range(num_blocks_h_chroma):
            for j in range(num_blocks_w_chroma):
                block = video["U"][f, i*block_size_chroma:(i+1)*block_size_chroma, j*block_size_chroma:(j+1)*block_size_chroma]
                
                block = intra_prediction(block)
                #block = dct(block)
                blocks["U"][f,numblock,:,:] = block
                numblock += 1
 
    #V-Blöcke
    for f in range(frames):
        numblock = 0
        for i in range(num_blocks_h_chroma):
            for j in range(num_blocks_w_chroma):
                block = video["V"][f, i*block_size_chroma:(i+1)*block_size_chroma, j*block_size_chroma:(j+1)*block_size_chroma]
                
                block = intra_prediction(block)
                #block = dct(block)
                blocks["V"][f,numblock,:,:] = block
                numblock += 1

    bitstream = blocks
    return bitstream


  
def vertical_intra_prediction(block):
    block_height, block_width = block.shape

    # Erstelle einen leeren Block für die Vorhersage
    prediction_block = np.zeros((block_height, block_width), dtype=np.uint8)

    # Iteriere über jede Spalte im Block
    for j in range(block_width):
        # Der Vorhersagewert wird aus dem oberen Nachbarpixel abgeleitet
        prediction_value = block[0, j]

        # Setze die Vorhersagewerte für die aktuelle Spalte im Block
        prediction_block[:, j] = prediction_value

    return prediction_block


def horizontal_intra_prediction(block):
    block_height, block_width = block.shape

    # Erstelle einen leeren Block für die Vorhersage
    prediction_block = np.zeros((block_height, block_width), dtype=np.uint8)

    # Iteriere über jede Zeile im Block
    for i in range(block_height):
        # Der Vorhersagewert wird aus dem linken Nachbarpixel abgeleitet
        prediction_value = block[i, 0]

        # Setze die Vorhersagewerte für die aktuelle Zeile im Block
        prediction_block[i, :] = prediction_value

    return prediction_block


def diagonal_intra_prediction(block):
    block_height, block_width = block.shape

    # Erstelle einen leeren Block für die Vorhersage
    prediction_block = np.zeros((block_height, block_width), dtype=np.uint8)

    # Iteriere über jede Zeile im Block
    for i in range(block_height):
        # Iteriere über jede Spalte im Block
        for j in range(block_width):
            # Der Vorhersagewert wird aus dem oberen linken Nachbarpixel abgeleitet
            if i == 0 or j == 0:
               #for prediction_value = block[0, 0]
               prediction_value = block[i,j]
            else:
                prediction_value = block[i-1, j-1]

            # Setze den Vorhersagewert für das aktuelle Pixel im Block
            prediction_block[i, j] = prediction_value

    return prediction_block



def intra_prediction(block):
    #ausführen der intra prediciton
    block = diagonal_intra_prediction(block)
    #block = horizontal_intra_prediction(block)
    #block = vertical_intra_prediction(block)
   
    return block


