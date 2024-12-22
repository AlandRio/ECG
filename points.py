import math as math
import numpy as np
import shared as shared

class Points:  # class for storing points
    def __init__(self, x_points=[], y_points=[],labels = []):
        self.x_points = x_points  # Original Points on the X-Axis in time domain 
        self.y_points = y_points  # Original Points on the Y-Axis in time domain
        self.labels = labels # Label such as S1 S2 S3


class Segment:
    def __init__(self,points = Points(),label = 0):
        self.points = points
        self.label = label



# ~ ~ ~ ~ Pre Processing ~ ~ ~ ~

# Removes DC Component
def removeMean(old_points=None):
    if old_points is None:
        old_points = []  # Create a default Points object if none is provided

    old_points_y_copy = np.copy(old_points)
    
    # Create a new Points object to store the transformed points
    new_points = []
    
    # Calculate the mean of the old points
    sum_points = sum(np.copy(old_points_y_copy)) 
    len_points = len(np.copy(old_points_y_copy))
    mean = sum_points/ len_points
    for y in old_points_y_copy:
        new_point = y - mean
        new_points.append(new_point)  # Subtract the mean and append to new_points
    return np.copy(new_points)


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


def applybutterworthBandpassFilter(numerator_coeffs, denominator_coeffs,input_points = Points()):
    """
    Applies the Butterworth bandpass filter to a given signal.

    Parameters:
        input_signal (ndarray): Input signal to be filtered.
        numerator_coeffs (ndarray): Numerator coefficients of the filter (b).
        denominator_coeffs (ndarray): Denominator coefficients of the filter (a).

    Returns:
        output_signal (ndarray): Filtered signal.
    """
    input_signal = np.array(input_points.y_points)
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
    output_points = Points(
        x_points= list(range(len(output_signal))),
        y_points= output_signal.tolist(),
        labels= input_points.labels
    )
    return output_points


# Normalizes wave between -1 and 1 for easier computation
def normalize(old_points = None):
    if old_points is None:
        old_points = []  # Create a default Points object if none is provided
    new_points = []

    max_point = max(old_points)
    min_point = min(old_points)

    for y in old_points:
        fraction = (y - min_point) / (max_point - min_point)
        new_points.append(2 * fraction - 1)
    
    return new_points


# Resamples the points for easier computation
def downSample(old_points = None, small_n = 4):
    if old_points is None:
        old_points = []  # Create a default Points object if none is provided
    
    points_y = np.copy(old_points)
    new_points = []

    for y in points_y[::small_n]:
        new_points.append(y)

    return new_points


# Segments the data for easier computation
def segment(old_points = Points()):
    heart_beat = 50
    number_segments = 4
    segment = number_segments * heart_beat

    segments = []
    y_length = len(old_points.y_points)
    iterations = math.ceil(y_length / segment) 

    for n in range(0, iterations):
        start = 0 + segment * n
        end = segment + segment * n
        new_points = Points()
        new_points.x_points = np.copy(old_points.x_points[start:end:1])
        new_points.y_points = np.copy(old_points.y_points[start:end:1])
        new_points.labels = np.copy(old_points.labels[start:end:1])
        L_1 = 0
        L_2 = 0
        L_3 = 0
        for label in new_points.labels:
            if label == 1:
                L_1 += 1
            elif label == 2:
                L_2 += 1
            elif label == 3:
                L_3 += 1
        label = 0
        if L_1 == max(L_1,L_2,L_3):
            label = 1
        if L_2 == max(L_1,L_2,L_3):
            label = 2
        if L_3 == max(L_1,L_2,L_3):
            label = 3
        segment_entry = Segment(new_points,label=label) 
        segments.append(segment_entry)
    
    return segments

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
def correlate(old_y_points = None):
    if old_y_points is None:
        old_y_points = []
    
    first = np.copy(old_y_points)
    second = np.copy(old_y_points)

    new_y = []
    denominator = sum(np.pow(first,2)) * sum(np.pow(second,2))
    denominator =  np.sqrt(denominator)/len(old_y_points)

    for j in range(len(old_y_points)):
        num_sum = 0
        for n in range(len(old_y_points)):
            num_sum += first[n] * second[n]
        second = shiftLeft(second)
        numerator = num_sum/len(first)
        new_point = numerator/denominator
        new_y.append(new_point)
    new_y = np.copy(fixPoints(new_y))
    return new_y


# Calculate DCT of points
def DCT(old_points = None):
    if old_points is None:
        old_points = []
    new_points = []
    samples = len(old_points)
    normalize = np.sqrt(2/samples)
    for k in range(samples):
        new_point = 0
        inside_cos = 0
        for n in range(samples):
            inside_cos = (np.pi/(4*samples)) * (2*n - 1) * (2*k-1)
            inside_cos = np.cos(inside_cos)
            new_point += old_points[n] * inside_cos
        new_point = round(new_point * normalize,9)
        new_points.append(new_point)
    
    return new_points



