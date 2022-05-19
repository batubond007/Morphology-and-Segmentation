"""
    Batuhan Çalışkan   2309805
    Alişan Yıldırım    2172161
"""

import numpy as np
import os
import cv2 as cv
from PIL import Image
from dahuffman import HuffmanCodec
import pickle
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


class JPEG:
    def __init__(self, h, w, im_h, im_w):
        self.block_codecs = np.empty((h, w), dtype=object)
        self.encoded_blocks = np.empty((h, w), dtype=object)
        self.downSampled_Cr = np.zeros((im_h//4, im_w//4))
        self.downSampled_Cb = np.zeros((im_h//4, im_w//4))

quantization_table = np.array([
    [1, 1, 1, 2, 4, 6, 10, 15],
    [1, 1, 1, 2, 4, 6, 10, 15],
    [1, 1, 1, 2, 4, 6, 10, 15],
    [2, 2, 2, 2, 4, 6, 10, 15],
    [4, 4, 4, 4, 4, 6, 10, 15],
    [6, 6, 6, 6, 6, 6, 10, 15],
    [10, 10, 10, 10, 10, 10, 10, 15],
    [15, 15, 15, 15, 15, 15, 15, 15]
])

quantization_table_2 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 13, 16, 24, 40, 57, 69, 56],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 103, 92],
    [49, 64, 78, 77, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

def down_sampler(Image):
    kernel = np.ones((4, 4))
    convolved = convolve2d(kernel, Image, mode='valid')
    a_downsampled = convolved[::4, ::4] / 16
    return a_downsampled

def up_sampler(Image, h, w):
    result = np.zeros((h*4, w*4))
    for x in range(h):
        x_start = x*4
        for y in range(w):
            y_start = y * 4
            result[x_start][y_start] = Image[x][y]
            result[x_start+1][y_start] = Image[x][y]
            result[x_start+2][y_start] = Image[x][y]
            result[x_start+3][y_start] = Image[x][y]

            result[x_start][y_start+1] = Image[x][y]
            result[x_start + 1][y_start+1] = Image[x][y]
            result[x_start + 2][y_start+1] = Image[x][y]
            result[x_start + 3][y_start+1] = Image[x][y]

            result[x_start][y_start+2] = Image[x][y]
            result[x_start + 1][y_start+2] = Image[x][y]
            result[x_start + 2][y_start+2] = Image[x][y]
            result[x_start + 3][y_start+2] = Image[x][y]

            result[x_start][y_start+3] = Image[x][y]
            result[x_start + 1][y_start+3] = Image[x][y]
            result[x_start + 2][y_start+3] = Image[x][y]
            result[x_start + 3][y_start+3] = Image[x][y]

    return result

def block_shifter(Arr, sign):
    for x in range(8):
        for y in range(8):
            Arr[x][y] += sign * 128.0
    return Arr


def matrixArrenger_3D(arr1, height, width):
    arr = np.zeros((height, width, 2))
    for x in range(height):
        for y in range(width):
            arr[x][y][0] = arr1[y][x]
            arr[x][y][1] = arr1[y][x]
    return arr


def gauss_2D(sigma, mu, height, width):
    x, y = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width))
    d = np.sqrt(x * x + y * y)
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    return g


def ideal_lowpass(height, width, r):
    arr = np.zeros((height, width))
    for h in range(height):
        for w in range(width):
            h_dif = abs(h - height / 2)
            w_dif = abs(w - width / 2)
            ratio = (h_dif ** 2 + w_dif ** 2) ** .5 / height
            if ratio < r:
                arr[h][w] = 1
    return arr


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def part1(input_img_path, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    Input_Image = cv.imread(input_img_path, cv.IMREAD_GRAYSCALE)
    Image_Height = Input_Image.shape[0]
    Image_Width = Input_Image.shape[1]

    gaussian_array = gauss_2D(.5, 0, Image_Height, Image_Width)
    gaussian_array = np.subtract(np.full((Image_Width, Image_Height), 1), gaussian_array)
    gaussian_array *= np.full((Image_Width, Image_Height), 255)
    gaussian_array = np.abs(gaussian_array)

    dft = cv.dft(np.float32(Input_Image), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    f_shift = dft_shift * matrixArrenger_3D(gaussian_array, Image_Height, Image_Width)

    f_ishift = np.fft.ifftshift(f_shift)
    img = cv.idft(f_ishift)
    img = cv.magnitude(img[:, :, 0], img[:, :, 1])

    cv.imwrite(output_path + "edges.png", img)


def enhance_3(path_to_3, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    Input_Image = cv.imread(path_to_3, cv.IMREAD_COLOR)
    Image_Height = Input_Image.shape[0]
    Image_Width = Input_Image.shape[1]

    gaussian_array = gauss_2D(.3, 0, Image_Height, Image_Width)
    gaussian_array = np.abs(gaussian_array)
    gaussian_array_3d = matrixArrenger_3D(gaussian_array, Image_Height, Image_Width)

    # low_pass = ideal_lowpass(Image_Height, Image_Width, .25)

    Red_Channel = Input_Image[:, :, 0]
    Green_Channel = Input_Image[:, :, 1]
    Blue_Channel = Input_Image[:, :, 2]

    spectrum = np.fft.fftshift(np.fft.fft2(Red_Channel))
    spectrum *= gaussian_array_3d[:, :, 0]
    # spectrum *= low_pass
    img_back = np.fft.ifft2(np.fft.ifftshift(spectrum))
    img_r = np.real(img_back)

    spectrum = np.fft.fftshift(np.fft.fft2(Green_Channel))
    spectrum *= gaussian_array_3d[:, :, 0]
    # spectrum *= low_pass
    img_back = np.fft.ifft2(np.fft.ifftshift(spectrum))
    img_g = np.real(img_back)

    spectrum = np.fft.fftshift(np.fft.fft2(Blue_Channel))
    spectrum *= gaussian_array_3d[:, :, 0]
    # spectrum *= low_pass
    img_back = np.fft.ifft2(np.fft.ifftshift(spectrum))
    img_b = np.real(img_back)

    Output_Image = np.zeros((Image_Height, Image_Width, 3))
    for x in range(0, Image_Width):
        for y in range(0, Image_Height):
            Output_Image[y][x][0] = img_r[y][x]
            Output_Image[y][x][1] = img_g[y][x]
            Output_Image[y][x][2] = img_b[y][x]

    cv.imwrite(output_path + "3.png", Output_Image)


def enhance_4(path_to_4, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    Input_Image = cv.imread(path_to_4, cv.IMREAD_COLOR)
    Image_Height = Input_Image.shape[0]
    Image_Width = Input_Image.shape[1]

    gaussian_array = gauss_2D(.5, 0, Image_Height, Image_Width)
    gaussian_array = np.abs(gaussian_array)
    gaussian_array_3d = matrixArrenger_3D(gaussian_array, Image_Height, Image_Width)

    low_pass = ideal_lowpass(Image_Height, Image_Width, .25)

    Red_Channel = Input_Image[:, :, 0]
    Green_Channel = Input_Image[:, :, 1]
    Blue_Channel = Input_Image[:, :, 2]

    spectrum = np.fft.fftshift(np.fft.fft2(Red_Channel))
    # spectrum *= gaussian_array_3d[:, :, 0]
    spectrum *= low_pass
    img_back = np.fft.ifft2(np.fft.ifftshift(spectrum))
    img_r = np.real(img_back)

    spectrum = np.fft.fftshift(np.fft.fft2(Green_Channel))
    # spectrum *= gaussian_array_3d[:, :, 0]
    spectrum *= low_pass
    img_back = np.fft.ifft2(np.fft.ifftshift(spectrum))
    img_g = np.real(img_back)

    spectrum = np.fft.fftshift(np.fft.fft2(Blue_Channel))
    # spectrum *= gaussian_array_3d[:, :, 0]
    spectrum *= low_pass
    img_back = np.fft.ifft2(np.fft.ifftshift(spectrum))
    img_b = np.real(img_back)

    Output_Image = np.zeros((Image_Height, Image_Width, 3))
    for x in range(0, Image_Width):
        for y in range(0, Image_Height):
            Output_Image[y][x][0] = img_r[y][x]
            Output_Image[y][x][1] = img_g[y][x]
            Output_Image[y][x][2] = img_b[y][x]

    cv.imwrite(output_path + "4.png", Output_Image)


def the2_write(input_img_path, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Image in YCbCr space
    img_ycbcr = np.array(Image.open(input_img_path).convert('YCbCr'))
    Image_Height = img_ycbcr.shape[0]
    Image_Width = img_ycbcr.shape[1]

    img_y = img_ycbcr[:, :, 0]
    img_Cb = img_ycbcr[:, :, 1]
    img_Cr = img_ycbcr[:, :, 2]

    ds_img_Cb = down_sampler(img_Cb)
    ds_img_Cr = down_sampler(img_Cr)

    height_block_count = Image_Height // 8
    width_block_count = Image_Width // 8

    block_codecs = np.empty((height_block_count, width_block_count), dtype=object)
    encoded_blocks = np.empty((height_block_count, width_block_count), dtype=object)

    for h in range(height_block_count):
        h_start = h * 8
        h_end = h_start + 8
        for w in range(width_block_count):
            w_start = w * 8
            w_end = w_start + 8

            block_y = np.float32(img_y[h_start:h_end, w_start:w_end])

            block_y_shift = block_shifter(block_y, -1)

            dct_block_y = cv.dct(block_y_shift)

            quantized_y = dct_block_y // quantization_table

            ordered_dct_block_y = zigzag(quantized_y).astype(int)

            codec = HuffmanCodec.from_data(ordered_dct_block_y.flatten())
            block_codecs[h][w] = codec

            encoded = codec.encode(ordered_dct_block_y)
            encoded_blocks[h][w] = encoded

    jpeg = JPEG(height_block_count, width_block_count, Image_Height, Image_Width)
    jpeg.block_codecs = block_codecs
    jpeg.encoded_blocks = encoded_blocks
    jpeg.downSampled_Cb = ds_img_Cb
    jpeg.downSampled_Cr = ds_img_Cr

    file = open(output_path + "jpeg.txt", 'wb')
    pickle.dump(jpeg, file)
    file.close()

    return output_path + "jpeg.txt"


def the2_read(input_img_path):
    file = open(input_img_path, 'rb')
    data_pickle = file.read()
    file.close()
    jpeg = pickle.loads(data_pickle)

    block_codecs = jpeg.block_codecs
    encoded_blocks = jpeg.encoded_blocks

    height_block_count = block_codecs.shape[0]
    width_block_count = block_codecs.shape[1]

    Output_Image = np.zeros((height_block_count * 8, width_block_count * 8, 3))

    for h in range(height_block_count):
        h_start = h * 8
        h_end = h_start + 8
        for w in range(width_block_count):
            w_start = w * 8
            w_end = w_start + 8

            decoded = block_codecs[h][w].decode(encoded_blocks[h][w])

            quantized = inverse_zigzag(decoded, 8, 8)

            dct_block_y = quantized * quantization_table

            block_y = np.array(cv.idct(dct_block_y)).astype(int)

            block_y_reshifted = block_shifter(block_y, 1)

            for x in range(8):
                for y in range(8):
                    Output_Image[h_start + x][w_start + y][0] = min(block_y_reshifted[x][y], 255)

    up_sampled_Cb = up_sampler(jpeg.downSampled_Cb, height_block_count * 2, width_block_count * 2)
    up_sampled_Cr = up_sampler(jpeg.downSampled_Cr, height_block_count * 2, width_block_count * 2)

    for x in range(height_block_count * 8):
        for y in range(width_block_count * 8):
            Output_Image[x][y][1] = up_sampled_Cb[x][y]
            Output_Image[x][y][2] = up_sampled_Cr[x][y]

    Output_Image_RGB = ycbcr2rgb(Output_Image)

    original_size = height_block_count * width_block_count * 64 * 24

    compressed_size = os.stat(input_img_path).st_size

    print(original_size, compressed_size, original_size / compressed_size)

    plt.imshow(Output_Image_RGB)
    plt.show()


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


def zigzag(input):
    # initializing the variables
    # ----------------------------------
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    vmax = input.shape[0]
    hmax = input.shape[1]

    # print(vmax ,hmax )

    i = 0

    output = np.zeros((vmax * hmax))
    # ----------------------------------

    while ((v < vmax) and (h < hmax)):

        if ((h + v) % 2) == 0:  # going up

            if (v == vmin):
                # print(1)
                output[i] = input[v, h]  # if we got to the first line

                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1

                i = i + 1

            elif ((h == hmax - 1) and (v < vmax)):  # if we got to the last column
                # print(2)
                output[i] = input[v, h]
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax - 1)):  # all other cases
                # print(3)
                output[i] = input[v, h]
                v = v - 1
                h = h + 1
                i = i + 1


        else:  # going down

            if ((v == vmax - 1) and (h <= hmax - 1)):  # if we got to the last line
                # print(4)
                output[i] = input[v, h]
                h = h + 1
                i = i + 1

            elif (h == hmin):  # if we got to the first column
                # print(5)
                output[i] = input[v, h]

                if (v == vmax - 1):
                    h = h + 1
                else:
                    v = v + 1

                i = i + 1

            elif ((v < vmax - 1) and (h > hmin)):  # all other cases
                # print(6)
                output[i] = input[v, h]
                v = v + 1
                h = h - 1
                i = i + 1

        if ((v == vmax - 1) and (h == hmax - 1)):  # bottom right element
            # print(7)
            output[i] = input[v, h]
            break

    # print ('v:',v,', h:',h,', i:',i)
    return output

def inverse_zigzag(input, vmax, hmax):
    # print input.shape

    # initializing the variables
    # ----------------------------------
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    output = np.zeros((vmax, hmax))

    i = 0
    # ----------------------------------

    while ((v < vmax) and (h < hmax)):
        # print ('v:',v,', h:',h,', i:',i)
        if ((h + v) % 2) == 0:  # going up

            if (v == vmin):
                # print(1)

                output[v, h] = input[i]  # if we got to the first line

                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1

                i = i + 1

            elif ((h == hmax - 1) and (v < vmax)):  # if we got to the last column
                # print(2)
                output[v, h] = input[i]
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax - 1)):  # all other cases
                # print(3)
                output[v, h] = input[i]
                v = v - 1
                h = h + 1
                i = i + 1


        else:  # going down

            if ((v == vmax - 1) and (h <= hmax - 1)):  # if we got to the last line
                # print(4)
                output[v, h] = input[i]
                h = h + 1
                i = i + 1

            elif (h == hmin):  # if we got to the first column
                # print(5)
                output[v, h] = input[i]
                if (v == vmax - 1):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1

            elif ((v < vmax - 1) and (h > hmin)):  # all other cases
                output[v, h] = input[i]
                v = v + 1
                h = h - 1
                i = i + 1

        if ((v == vmax - 1) and (h == hmax - 1)):  # bottom right element
            # print(7)
            output[v, h] = input[i]
            break

    return output

