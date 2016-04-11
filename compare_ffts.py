#!/usr/bin/env python

# make random sample
# run fft over all of them
# ensure differ by small amount
def compareFFTs(adapters, iterations, sample_size=64):
    import numpy as np
    for i in xrange(0, iterations):
        data = np.random.rand(sample_size)
        ffts = { name: np.array(adapter.fft(list(data))) for name, adapter in adapters.items() }
        fft_items = ffts.items()
        for fft_index, item in enumerate(fft_items):
            if fft_index == 0:
                continue

            prev_fft = fft_items[fft_index-1][1]
            prev_name = fft_items[fft_index-1][0]
            fft = item[1]
            name = item[0]
            if not np.allclose(prev_fft[i], fft[i]):
                print "ffts differ: {} vs {}".format(prev_name, name)
                print "  {}".format(fft - prev_fft)

import fftw_fft, opencv_fft
iterations = 2
sample_size = 64
adapters = {
    'fftw': fftw_fft.FFTWPythonAdapter(sample_size),
    'opencv': opencv_fft.OpenCVFFTPythonAdapter(sample_size),
}
compareFFTs(adapters, iterations, sample_size);
