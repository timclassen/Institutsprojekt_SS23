import numpy as np
import math
from scipy.fftpack import idct

def decoder(bitstream):
    print("Decoding video...")

    frames = bitstream["Y"].shape[0]
    num_blocks_perframe_luma = bitstream["Y"].shape[1]
    block_size_luma = bitstream["Y"].shape[2]
    #Berechnung der Videogröße gilt nur für Videos mit "quadratischer Form"
    height_luma = int(math.sqrt(num_blocks_perframe_luma) * block_size_luma)
    width_luma = int(math.sqrt(num_blocks_perframe_luma) * block_size_luma)

    num_blocks_perframe_chroma = bitstream["U"].shape[1]
    block_size_chroma = bitstream["U"].shape[2]
    height_chroma = int(math.sqrt(num_blocks_perframe_chroma) * block_size_chroma)
    width_chroma = int(math.sqrt(num_blocks_perframe_chroma) * block_size_chroma)

    temp_video = {"Y": np.zeros((frames, height_luma, width_luma), dtype=np.uint8),
             "U": np.zeros((frames, height_chroma, width_chroma), dtype=np.uint8),
             "V": np.zeros((frames, height_chroma, width_chroma), dtype=np.uint8)}

    
    #Laden der Blöcke der einzelnen channels an ihre zugehörige Position im jeweiligem Frame
    for f in range(frames):
        numblock = 0
        for i in range(int(math.sqrt(num_blocks_perframe_luma))):
            for j in range(int(math.sqrt(num_blocks_perframe_luma))):

                block = (bitstream["Y"][f,numblock,:,:])
                #block = idct(block)
                temp_video["Y"][f, i*block_size_luma:(i+1)*block_size_luma, j*block_size_luma:(j+1)*block_size_luma] = block

                block = (bitstream["U"][f,numblock,:,:])
                #block = idct(block)
                temp_video["U"][f, i*block_size_chroma:(i+1)*block_size_chroma, j*block_size_chroma:(j+1)*block_size_chroma] = block

                block = (bitstream["V"][f,numblock,:,:])
                #block = idct(block)
                temp_video["V"][f, i*block_size_chroma:(i+1)*block_size_chroma, j*block_size_chroma:(j+1)*block_size_chroma] = block
                
                numblock += 1
  
    video = temp_video
    return video

