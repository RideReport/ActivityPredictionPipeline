#!/usr/bin/env python

# make random sample
# run fft over all of them
# ensure differ by small amount
def compareFFTs(adapters, iterations, sample_size=64):
    import numpy as np
    for i in xrange(0, iterations):
        data = np.array(np.random.rand(sample_size) * 1e2, dtype=float)
        ffts = { name: np.array(adapter.fft(list(data)), dtype=np.float32) for name, adapter in adapters.items() }
        fft_items = ffts.items()
        for fft_index, item in enumerate(fft_items):
            if fft_index == 0:
                continue

            prev_fft = fft_items[fft_index-1][1]
            prev_name = fft_items[fft_index-1][0]
            fft = item[1]
            name = item[0]
            close = np.isclose(prev_fft[:sample_size/2], fft[:sample_size/2], rtol=1e-3)
            if not np.allclose(prev_fft[:sample_size/2], fft[:sample_size/2], rtol=1e-3):
                print "ffts differ: {} vs {}".format(prev_name, name)
                print "  {}".format((fft - prev_fft)[:sample_size])
                print "  {}".format(fft[:sample_size])
                print "  {}".format(prev_fft[~close[:sample_size]])
                print "  {}".format(fft[~close[:sample_size]])
                first_bad = np.argmax(~close[:sample_size])
                print "  {}".format(first_bad)


import fftw_fft, opencv_fft
iterations = 10000
sample_size = 64
adapters = {
    'fftw': fftw_fft.FFTWPythonAdapter(sample_size),
    'opencv': opencv_fft.OpenCVFFTPythonAdapter(sample_size),
}
compareFFTs(adapters, iterations, sample_size);
