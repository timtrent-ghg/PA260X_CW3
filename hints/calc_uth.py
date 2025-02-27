def calc_uth(T6_7, p0):
    """ calculate the clear-sky UTH for the NOAA HIRS CDR
    inputs: T6_7  -> HIRS WV channel BTS @ 6.7 microns [K]
            p0    -> normalised 240 K isotherm

    outputs: UTH  -> clear-sky UTH [%]
    """
    # define linear fit intercept
    a = 31.5 
    # define linear fit gradient
    b = -0.115
    # calculate clear-sky UTH, assuming cos theta = 1. Note for exponentials
    # you can use the np.exp() method
    uth =  (1/p0)*np.exp(a+b*T6_7)
    # return the clear sky UTH
    return uth
