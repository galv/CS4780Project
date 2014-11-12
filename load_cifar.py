def load_cifar(source):
    source = source.astype(np.uint8)
    if not cv2image:
        source = source.transpose([1,0]) #we expect width/height but use col/row
        self._bitmap = cv.CreateImage((source.shape[1], source.shape[0]), cv.IPL_DEPTH_8U, 3)
        channel = cv.CreateImageHeader((source.shape[1], source.shape[0]), cv.IPL_DEPTH_8U, 1)
        #initialize an empty channel bitmap
        cv.SetData(channel, source.tostring(),
                source.dtype.itemsize * source.shape[1])
        cv.Merge(channel, channel, channel, None, self._bitmap)
        self._colorSpace = ColorSpace.BGR
     
