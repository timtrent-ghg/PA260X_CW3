import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

##################################################################################################
def read_csv(filename):
    """ read in the csv files we previously made of UTH anomaly time series.
    inputs: filename -> full path to file containing our data

    ouputs: results -> python dictionary containing our anomaly time series.
    """
    results = {}
    data = np.loadtxt(filename, skiprows=1, delimiter=",",dtype='<f4')
    header = np.genfromtxt(filename, delimiter=',', dtype=str, max_rows=1)
    for ii, varname in enumerate(header):
        results[varname] = data[:,ii]
    return results
    
##################################################################################################
def simple_moving_average(yvals, width):
    """ compute the moving average of a time series with a user defined sliding window.
    inputs: yvals -> time series on regular time steps
            width -> width of sliding window (n time steps)
    output: sommothed time series
    """
    return np.convolve(yvals, np.ones(width), 'same')/width
    
##################################################################################################
def estimate_coef(x, y):
    """ python implimentation of linear regression.
    Inputs: x -> 1D array (e.g. fractional year)
            y -> 1D array (e.g. UTH anomaly)

    Outputs: coeffs -> tuple containing the intercept and gradient
    """
    # define number of observations/points
    n = np.size(x)
    
    # calculate the mean of the x and y vectors
    m_x = np.mean(x)
    m_y = np.mean(y)
    
    # calculate cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    
    # calculate regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    
    return (b_0, b_1)

##################################################################################################
def calculate_probDensFunc(yvals, ymin, ymax,nbins=100):
    """ calculate a PDF from the array yvals using scipy.stats norm module
    inputs: yvals -> 1d array of values to be used in PDF calculation
            ymin  -> minimum value for range of yvals PDF is calculated
            ymax  -> maximum value for range of yvals PDF is calculated
            nbins -> number of bins for which PDF is calculated between ymin  and ymax
            
    outputs: xvals -> array of values over which the PDF was calculated (defined by ymin, ymax,nbins)
             PDF   -> array containing PDF values
    """
    # define xvals
    xvals = np.linspace(ymin,ymax,nbins)

    # calculate mean and standard deviation of yvals
    mu = np.mean(yvals)
    std = np.std(yvals)

    # define an empty array to hold PDF values. Here we use the size method to tell the code how many elements 
    # are in this new array and fill each entry with a default value NaN (Not a Number)
    PDF = np.full(xvals.size, np.nan)

    # loop over each value in x and calculate the corresponding PDF value. Because xvals is an array merans in 
    # Python it is iterable (i.e. we can loop over the contents) and by wrapping it in the enumerate function
    # we also get the index of the value (e.g. the firts value in xvals could be -2, therefore x=-2 and ii=0).
    for ii, x in enumerate(xvals):
        PDF[ii] = norm.pdf(x, loc=mu, scale=std)

    # return the results
    return PDF, xvals

##################################################################################################
def plot_delta_tas_probDensFunc(xvals, pre_glb_pdf, pre_lnd_pdf, pre_ocn_pdf, aft_glb_pdf, aft_lnd_pdf, aft_ocn_pdf, output=False):
    """ function to plot PDF distributions
    inputs: xvals         ->
            pre_glb_pdf
            pre_lnd_pdf
            pre_ocn_pdf
            aft_glb_pdf
            aft_lnd_pdf
            aft_ocn_pdf
            
            
    output:
    """
    plt.figure(figsize=(4,8),dpi=200)
    plt.subplot(211)
    plt.plot(xvals, pre_glb_pdf,lw=2,color='#A8AAB7',label='Global PDF')
    plt.plot(xvals, pre_lnd_pdf,'--',lw=2,color="#00A7B5",label='Land PDF')
    plt.plot(xvals, pre_ocn_pdf,'-.',lw=2,color="#F84C4F",label='Ocean PDF')
    plt.xlim(-3,2)
    plt.ylim(0,1.5)
    plt.legend(loc=2,fontsize=8)
    plt.ylabel("PDF",fontsize=12)
    plt.xlabel(r"$\Delta$UTH [%]",fontsize=12)
    plt.title("a) 1978-2000")
    
    plt.subplot(212)
    plt.plot(xvals, aft_glb_pdf,lw=2,color='#A8AAB7',label='Global PDF')
    plt.plot(xvals, aft_lnd_pdf,'--',lw=2,color="#00A7B5",label='Land PDF')
    plt.plot(xvals, aft_ocn_pdf,'-.',lw=2,color="#F84C4F",label='Ocean PDF')
    plt.xlim(-3,2)
    plt.ylim(0,1.5)
    plt.legend(loc=2,fontsize=8)
    plt.ylabel("PDF",fontsize=12)
    plt.xlabel(r"$\Delta$UTH [%]",fontsize=12)
    plt.title("b) 2001-2024")
        
    plt.tight_layout()
    

    # finally we test to see whether the figure will be saved as an .pdf file or rather 
    # just plotted within the notebook.
    if output == True:
        # save the figure with a descriptive name
        plt.savefig("UTH_PDF_land_ocean_global_analysis.pdf")
        plt.close()
    else:
        # just plot to screen
        pass
        
##################################################################################################
# read in the data
data = read_csv("uth_anomoly_timeseries.csv")

##################################################################################################
# apply a simple 12 month moving average filter to our time series.
anom_sma_glb = simple_moving_average(data['Global'],12)
anom_sma_lnd = simple_moving_average(data['Land'],12)
anom_sma_ocn = simple_moving_average(data['Ocean'],12)

##################################################################################################
# regress smoothed time series against fractional year and return linear fit coefficients
glb_coeffs = estimate_coef(data['Frac_Year'], anom_sma_glb)
lnd_coeffs = estimate_coef(data['Frac_Year'], anom_sma_lnd)
ocn_coeffs = estimate_coef(data['Frac_Year'], anom_sma_ocn)

##################################################################################################
# define end of row for LaTeX
endofrow=r"\\"
units=r"\%/dec" 
# create a list of the outputs we want to write to file
table = [r"\begin{table}[t]",
         r"    \centering",
         r"    \caption{Decal trends in HIRS UTH for land, ocean, and cobimed (global) surfaces between $\pm$30^{\circ}.}",
         r"    \begin{tabular}{l c c}",
         r"        \toprule",
         f"        Region & Trend ({units}){endofrow}",
         r"        \midrule",
         f"        Global & {glb_coeffs[1]*120:0.2f}{endofrow}",
         f"        Land & {lnd_coeffs[1]*120:0.2f}{endofrow}",
         f"        Ocean & {ocn_coeffs[1]*120:0.2f}{endofrow}",
         r"        \bottomrule",
         r"    \end{tabular}",
         r"    \label{tab_uth_trends}",
         r"\end{table}"]

with open("trend_results.txt", "w") as fobj:
    for line in table:
        fobj.write(f"{line}\n")

##################################################################################################
# calculate PDFs of UTH anonaly prior to 2001 and after 2001 for global, land and ocean results
pre2000 = np.where(data['Frac_Year'] < 2001)
aft2000 = np.where(data['Frac_Year'] >= 2001)

pre_glb_pdf, xvals = calculate_probDensFunc(anom_sma_glb[pre2000], -5, 3, nbins=80)
pre_lnd_pdf, xvals = calculate_probDensFunc(anom_sma_lnd[pre2000], -5, 3, nbins=80)
pre_ocn_pdf, xvals = calculate_probDensFunc(anom_sma_ocn[pre2000], -5, 3, nbins=80)

aft_glb_pdf, xvals = calculate_probDensFunc(anom_sma_glb[aft2000], -5, 3, nbins=80)
aft_lnd_pdf, xvals = calculate_probDensFunc(anom_sma_lnd[aft2000], -5, 3, nbins=80)
aft_ocn_pdf, xvals = calculate_probDensFunc(anom_sma_ocn[aft2000], -5, 3, nbins=80)

plot_delta_tas_probDensFunc(xvals, pre_glb_pdf, pre_lnd_pdf, pre_ocn_pdf, aft_glb_pdf, aft_lnd_pdf, aft_ocn_pdf, output=False)