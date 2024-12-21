import math as math
import numpy as np
import shared as shared

class Points:  # class for storing points
    def __init__(self, x_points=[], y_points=[],label = "null"):
        self.x_points = x_points  # Original Points on the X-Axis in time domain 
        self.y_points = y_points  # Original Points on the Y-Axis in time domain
        self.label = str(label)

# ~ ~ ~ ~ Pre Processing ~ ~ ~ ~

# Removes DC Component
def removeMean(old_points= Points()):
    new_points = Points() # Makes a new object to store the new points
    mean = sum(old_points.y_points)/len(old_points.y_points)  # Calculates the mean to remove it from each point
    new_points.x_points = old_points.x_points # IDX is unchanged so we copy it into new points
    for x in range(len(old_points.x_points)): # For loop goes over each point in Y and removes the mean from it
        new_points.y_points.append(old_points.y_points[x] - mean)
    return new_points

# Convolves points
def convolve(first_points = Points(),second_points = Points()):
    new_y = [] # Temporary list 
    min_point = int(first_points.x_points[0] + second_points.x_points[0]) # Calculates the minimum of the range for the indexes of convolved points
    max_point = int(first_points.x_points[-1] + second_points.x_points[-1] + 1) # Calcualtes the maximum of the range for the indexes of convolved points
    first_length = len(first_points.y_points) 
    second_length = len(second_points.y_points)
    for i in range( first_length + second_length  - 1): 
        conv_sum = 0 # Initializes the sum by 0 to add on it
        for j in range(first_length):
                if i-j >= 0 and i-j < second_length: # Checks if the index is viable to avoid IndexError
                    conv_sum = conv_sum + first_points.y_points[j] * second_points.y_points[i - j]
        new_y.append(conv_sum)
    conv_points = Points()
    conv_points.x_points = list(range(min_point,max_point)) # Assigns a list of a range of numbers from minimum point to maxmimum point
    conv_points.y_points = new_y 
    return conv_points

# Returns the Coeficents, Dont forget to convolve with original signal with convolvePoints()
# def butterworth(samp_freq,trans_band,stop_atten,cut_off_1,cut_off_2):
#     # Checks if stop attentuation is outside of range
#     if stop_atten < 0 or stop_atten > 74:
#         print("Error Stop attentuation must be between 0-74")
#         return
#
#     # Calculates the normalized Trans Band
#     normalized_trans = trans_band/samp_freq
#     # Calculates the cut off frequencies for practical performance
#     cut_off_1 = cut_off_1 - trans_band/2
#     cut_off_2 = cut_off_2 + trans_band/2
#     # Normalizes cut off frequencies
#     cut_off_2 = cut_off_2/samp_freq
#     cut_off_1 = cut_off_1/samp_freq
#     # Calcualtes the omega frequencies
#     omega_1 = cut_off_1*2*math.pi
#     omega_2 = cut_off_2*2*math.pi
#
#     # Initialize variables used in filter
#     window = 0
#     w_n_0 = 0
#     big_N = 0
#
#
#
#     # Chooses window based on stop attentuation
#     if stop_atten <= 21: # Rectangular
#         window = 0
#         big_N = 0.9/normalized_trans
#         w_n_0 = 1
#     elif stop_atten <= 44: # Hanning
#         window = 1
#         big_N = 3.1/normalized_trans
#         w_n_0 = 0.5 + 0.5*math.cos(0)
#     elif stop_atten <= 53: # Hamming
#         window = 2
#         big_N = 3.3/normalized_trans
#         w_n_0 = 0.54+ 0.46*math.cos(0)
#     elif stop_atten <= 74: # Blackman
#         window = 3
#         big_N = 5.5/normalized_trans
#         w_n_0 = 0.42 + 0.5*math.cos(0)+0.08*math.cos(0)
#
#     # Makes sure Big_N is an ODD number so we can use Type 1 Filter
#     big_N = math.ceil(big_N)
#     if big_N % 2 == 0:
#         big_N = big_N + 1
#
#     # Initializes Coefficents list
#     h_coef = []
#     # Calculates the first element
#     h_d_0 = 2*(cut_off_2-cut_off_1)
#     # Appends the first element to the list of coefficents
#     h_coef.append(h_d_0 * w_n_0)
#
#     # Loop to calculate half of the coefficents as it is symmetric and we can just copy it
#     for n in range(1,math.ceil(big_N/2)):
#         h_d = 0
#         w_n = 0
#         h_d = 2*cut_off_2*math.sin(n*omega_2)/(n*omega_2)-2*cut_off_1*math.sin(n*omega_1)/(n*omega_1)
#         if window == 0:
#             w_n = 1
#         elif window == 1:
#             w_n = 0.5 + 0.5*math.cos(2*math.pi*n/big_N)
#         elif window == 2:
#             w_n = 0.54 + 0.46*math.cos(2*math.pi*n/big_N)
#         elif window == 3:
#             w_inside_cos = math.pi*n/(big_N-1)
#             w_n_param_number_1 = 0.5 * (math.cos(2*w_inside_cos))
#             w_n_param_number_2 = 0.08 * (math.cos(4*w_inside_cos))
#             w_n = 0.42 + w_n_param_number_1 + w_n_param_number_2
#         h_n = h_d * w_n
#         h_coef.append(h_n)
#
#     # Calculates the range from 0 to length of the coefficents (Which is half of the lenght + 1 since we havent copied the other side)
#     h_coef_idx = np.arange(0,len(h_coef)).tolist()
#     # Reverses the coefficents so we can copy them for symmetry
#     h_coef.reverse()
#     h_coef_idx.reverse()
#     # Copies the reversed lists
#     reverse_h_coef = np.copy(h_coef)
#     reverse_h_coef_idx = np.multiply(np.copy(h_coef_idx), -1) # Multiplies by negative so that the indexes are -ve
#     # Reverses the list again to undo the original reverse
#     h_coef_idx.reverse()
#     h_coef.reverse()
#     # Pops the first elements which at index 0 because it is repeated twice as the end of the copied list and the original list
#     h_coef.pop(0)
#     h_coef_idx.pop(0)
#
#     # Concats the copied reversed list + the original list
#     coef = Points()
#     coef.x_points = np.concatenate((reverse_h_coef,h_coef),axis=0).tolist()
#     coef.y_points = np.concatenate((reverse_h_coef_idx,h_coef_idx),axis=0).tolist()
#     return coef
def customButter(order , low_normal_cut_off , high_normal_cut_off):
    """
    Designs a Butterworth bandpass filter manually.

    Parameters:
        order (int): Order of the Butterworth filter.
        low_normal_cut_off (float): Normalized lower cutoff frequency (0 < low_normal_cut_off < 1).
        high_normal_cut_off (float): Normalized higher cutoff frequency (0 < high_normal_cut_off < 1).

    Returns:
        b (ndarray): Numerator coefficients of the filter.
        a (ndarray): Denominator coefficients of the filter.
    """
    # Check input validity
    if not (0 < low_normal_cut_off < high_normal_cut_off < 1):
        raise ValueError("Cutoff frequencies must satisfy 0 < low < high < 1.")

    # Pre-warp cutoff frequencies for bilinear transform
    warped_low_cutoff = 2 * np.tan(np.pi * low_normal_cut_off)
    warped_high_cutoff = 2 * np.tan(np.pi * high_normal_cut_off)
    warped_center = np.sqrt(warped_low_cutoff * warped_high_cutoff)  # Center frequency
    warped_bandwidth = warped_high_cutoff - warped_low_cutoff       # Bandwidth

    # Butterworth poles (analog prototype)
    poles = []
    for k in range(1, order + 1):
        angle = np.pi * (2 * k - 1) / (2 * order)
        pole = -np.sin(angle) + 1j * np.cos(angle)
        poles.append(pole)

    # Frequency transformation to bandpass
    bandpass_poles = []
    for pole in poles:
        s = pole * warped_bandwidth / 2
        bandpass_poles.append(s + np.sqrt(s ** 2 - warped_center ** 2))
        bandpass_poles.append(s - np.sqrt(s ** 2 - warped_center ** 2))

    # Bilinear transform to digital domain
    digital_poles = [(2 + p) / (2 - p) for p in bandpass_poles]
    digital_poles = np.clip(digital_poles, -1, 1)  # Ensure poles are in the unit circle

    # Calculate coefficients from poles
    a = np.poly(digital_poles)   # Denominator coefficients
    b = np.poly(np.zeros(len(digital_poles)))  # Numerator coefficients

    # Normalize gain at center frequency
    center_freq_normalized = (low_normal_cut_off + high_normal_cut_off) / 2
    gain = np.abs(np.polyval(b, np.exp(-1j * 2 * np.pi * center_freq_normalized)) /
                  np.polyval(a, np.exp(-1j * 2 * np.pi * center_freq_normalized)))
    b /= gain
    return b, a


def butterworthBandpassFilter(samp_rate,low_cut_off,high_cut_off,order=1):
    nyqs = 1 / 2 * samp_rate
    low_normal_cut_off = low_cut_off / nyqs
    high_normal_cut_off = high_cut_off / nyqs
    b , a = customButter(order , low_normal_cut_off, high_normal_cut_off,)
    return b,a
def applybutterworthBandpassFilter(input_signal, numerator_coeffs, denominator_coeffs):
    """
    Applies the Butterworth bandpass filter to a given signal.

    Parameters:
        input_signal (ndarray): Input signal to be filtered.
        numerator_coeffs (ndarray): Numerator coefficients of the filter (b).
        denominator_coeffs (ndarray): Denominator coefficients of the filter (a).

    Returns:
        output_signal (ndarray): Filtered signal.
    """
    # Initialize the output signal array
    output_signal = np.zeros_like(input_signal)

    # Apply the difference equation
    for sample_index in range(len(input_signal)):
        # Apply b[0] directly to the current input sample
        current_output = numerator_coeffs[0] * input_signal[sample_index]

        # Add contributions from remaining numerator coefficients
        current_output += sum(
            numerator_coeffs[coeff_index] * input_signal[sample_index - coeff_index]
            for coeff_index in range(1, len(numerator_coeffs))
            if sample_index - coeff_index >= 0
        )

        # Subtract contributions from denominator coefficients (skipping a[0])
        current_output -= sum(
            denominator_coeffs[coeff_index] * output_signal[sample_index - coeff_index]
            for coeff_index in range(1, len(denominator_coeffs))
            if sample_index - coeff_index >= 0
        )

        # Assign the calculated output to the output signal array
        output_signal[sample_index] = current_output

    return output_signal
# Normalizes wave between -1 and 1 for easier computation
def normalize(old_points = Points()):
    new_points = Points()

    max_point = max(old_points.y_points)
    min_point = min(old_points.y_points)

    for x in range(new_points.samples):
        new_points.x_points[x] = (old_points.x_points[x])
        fraction = (old_points.y_points[x] - min_point) / (max_point - min_point)
        new_points.y_points.append(2 * fraction - 1)
    
    return new_points


# Resamples the points for easier computation
def downSample(old_points = Points()):
    small_n = 4
    points_y = old_points.y_points
    new_points = Points()
    for y in points_y[::small_n]:
        new_points.y_points.append(float(y))
    new_points.x_points = list(range(0, len(new_points.y_points)))
    return new_points


# Segments the data for easier computation
def segment(old_points = Points()):
    heart_beat = 200
    number_segments = 4
    segment = number_segments * heart_beat

    new_points_list = []
    y_length = len(old_points.y_points)
    iterations = math.ceil(y_length / segment) 

    for n in range(0, iterations):
        start = 0 + segment * n
        end = segment + segment * n
        new_points = Points()
        new_points.x_points = old_points.x_points[start:end:1]
        new_points.y_points = old_points.y_points[start:end:1]
        new_points_list.append(new_points)
    
    return new_points_list

# ~ ~ ~ ~ Feature Extraction ~ ~ ~ ~ 

# Shifts the signal to the left
def shiftLeft(arr = []):
    new_points = []
    for x in range(len(arr)):
        new_points.append(arr[x-1])
    return new_points


# Fixes the points they're reversed beyond index 0 for some reason
def fixPoints(arr = []):
    new_points = []
    new_points.append(round(float(arr[0]),3))
    for x in range (1, len(arr)):
        new_points.append(round(float(arr[0-x]),3))
    return new_points
    

# Auto Correlation
def correlate(old_points = Points()):
    first = old_points
    second = old_points
    new_points = Points()
    denominator = sum(np.pow(first.y_points,2)) * sum(np.pow(second.y_points,2))
    denominator =  np.sqrt(denominator)/len(first.x_points)
    for j in range(len(second.x_points)):
        num_sum = 0
        for n in range(len(second.x_points)):
            num_sum += first.y_points[n] * second.y_points[n]
        second.y_points = shiftLeft(second.y_points)
        numerator = num_sum/len(first.x_points)
        new_point = numerator/denominator
        new_points.append(new_point)
    new_points.x_points = first.x_points
    new_points.y_points = fixPoints(new_points.y_points)
    print(f"{new_points.y_points}")
    return new_points


# Calculate DCT of points
def DCT(old_points = Points()):
    new_points = Points()
    samples = len(old_points.x_points)
    normalize = np.sqrt(2/samples)
    for k in range(samples):
        new_point = 0
        inside_cos = 0
        for n in range(samples):
            inside_cos = (np.pi/(4*samples)) * (2*n - 1) * (2*k-1)
            inside_cos = np.cos(inside_cos)
            new_point += old_points.y_points[n] * inside_cos
        new_point = round(new_point * normalize,9)
        new_points.y_points.append(new_point)
    new_points.x_points = old_points.x_points
    return new_points



