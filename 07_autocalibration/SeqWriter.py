# (c) 2020-04-03 Martin Matousek
# Last change: $Date$
#              $Revision$

import cv2
import numpy as np
import os.path

class SeqWriter:

    def __init__( this, filename ):
        this.video = None
        this.decimate = 1
        this.filename = filename
        this.frameno = 0
        _, ext = os.path.splitext( filename )

        if ext == '.avi':
            this.video = -1
            return

        raise Exception( 'Unhandled file type:' + filename )

    def Close( this ):
        if this.video != None:
            this.video.release()
            this.video = None

    def Write( this, frame ):
        if this.decimate > 1:
            d = int( np.ceil( this.decimate ) )

            nframe = 0.0
            for i in range( 0, d ):
                for j in range( 0, d ):
                    nframe = nframe + frame[i:-d+i:d, j:-d+j:d, : ].astype( float )

            frame = nframe / float( d * d )
            frame = frame.astype( np.uint8 )

        if this.video == -1:
            this.video = cv2.VideoWriter( this.filename,
                                          cv2.VideoWriter_fourcc('M','J','P','G'),
                                          25, 
                                          ( frame.shape[1], frame.shape[0] ) )
        
        this.frameno += 1

        if not this.video is None:
            frame_bgr = np.empty_like( frame );
            frame_bgr[ :, :, 0 ] = frame[ :, :, 2 ]
            frame_bgr[ :, :, 1 ] = frame[ :, :, 1 ]
            frame_bgr[ :, :, 2 ] = frame[ :, :, 0 ]
        
            this.video.write( frame_bgr )
