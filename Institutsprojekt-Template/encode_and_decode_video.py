# Please write your code for encoding and decoding the video
from yuv_io import read_yuv_video, write_yuv_video
from encoder import encoder
from decoder import decoder
from bitstream_io import write_bitstream, read_bitstream
from psnr import psnr_yuv
import os


data_path = "/home/staff/classen/Teaching/Institutsprojekt-Template/tmp/"


def encode_and_decode_video(yuv_video_path):
    '''
        Encodes and decodes the video given video. Have a look in the data folder for options of videos to encode. The coded stream and the decoded video is stored in the tmp folder. If you like, you can always ask me for more/larger videos to experiment or any kind of help ;) 
    '''
    coded_video_path = data_path + "VideoStream.svc"

    # Read the uncompressed video
    originalVideo = read_yuv_video(yuv_video_path)

    # Encode the video
    bitstream = encoder(originalVideo)
    write_bitstream(coded_video_path, bitstream, debug=True)
    
    # Decode the video
    bitstream = read_bitstream(coded_video_path, debug=True)
    decodedVideo = decoder(bitstream)

    # Write the reonstructed video
    write_yuv_video(decodedVideo, data_path + "DecodedVid.yuv")
    
    # Calculate statistics
    originalVideoSize = os.path.getsize(yuv_video_path)
    bitstreamSize = os.path.getsize(data_path + "VideoStream.svc")

    # Print statistics
    print("\n\n---------- Results ----------")
    print("Size of original video:", originalVideoSize)
    print("Size of bitstream:", bitstreamSize)
    print("Compression ratio:", originalVideoSize / bitstreamSize)
    print("PSNR:", psnr_yuv(originalVideo, decodedVideo))


encode_and_decode_video("/home/staff/classen/Teaching/Institutsprojekt-Template/data/ArenaOfValor_384x384_60_8bit_420.yuv")