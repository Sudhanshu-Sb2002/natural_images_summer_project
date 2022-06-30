import numpy as np
import mat73
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def input_gammas(path):
    rawdata = mat73.loadmat(path)
    st = rawdata['stPwrLfpTex'].astype(np.float32)
    bl = rawdata['blPwrLfpTex'].astype(np.float32)
    freq = rawdata['freqVals'].astype(np.float32)
    electrode_names = rawdata['LFPElectrodes'].astype(np.int64)

    st = np.transpose(st, (1, 0, 2))
    # images X Electrodes X Frequencies
    return st, bl, freq, electrode_names


def input_images(path, numpy_load=False):
    if numpy_load:
        rawdata = np.load(path)
        data = rawdata["masked_images"]
        return data
    else:
        rawdata = mat73.loadmat(path)
        data = (rawdata["masked_images"] * 255).astype(np.uint8)
        return data


def change_in_power(bl, st):
    delta_power = np.zeros(shape=st.shape)
    for i in range(st.shape[0]):
        delta_power[i] = np.divide(st[i], bl)
    return delta_power


def change_in_gamma(bl, st, freq):
    delta_power = change_in_power(bl, st)
    indices = np.logical_and(freq >= 30, freq <= 80)
    del_gamma = np.zeros(shape=(delta_power.shape[0], delta_power.shape[1]))
    for i in range(delta_power.shape[0]):
        for j in range(delta_power.shape[1]):
            del_gamma[i][j] = np.sum(delta_power[i][j][indices])
            '''plt.plot(freq, delta_power[i][j])
            plt.xlim(0, 100)
            plt.show()'''
    return del_gamma

def mini_rsquare_visualiser(x_array, y, x_label_array,ylabel,suptit):
    fig, ax = plt.subplots(3)
    fig.set_size_inches(6, 7)
    fig.set_dpi = 100
    fig.suptitle(suptit)
    for i in range(len(x_array)):
        ax[i].scatter(x_array[i], y[i])
        corr=round(np.corrcoef(x_array[i], y[i])[0,1],2)
        ax[i].set_title(x_label_array[i]+" Correlation: "+str(corr))
        ax[i].grid(True)
    plt.show()

def main():
    path1 = "D:\\OneDrive - Indian Institute of Science\\5th Sem\\Summer project\\Data\\simple_data_v1\\textureColorStimuliPowerSp_alpaH_st_250_500_bl_-250_0_summerProject.mat"
    path2 = "D:\\OneDrive - Indian Institute of Science\\5th Sem\\Summer project\\Data\\simple_data_v1\\RGB_texture_images_2.mat"
    path3 = "Compressed_images.npz"
    st, bl, freq, electrode_names = input_gammas(path1)
    delta_power = change_in_power(bl, st)
    delta_gamma = change_in_gamma(bl, st, freq)
    images = input_images(path3, numpy_load=True)

    delta_gamma_flat = np.resize(delta_gamma, new_shape=(delta_gamma.shape[0] * delta_gamma.shape[1]))
    images_flat = np.resize(images, new_shape=(
    images.shape[0] * images.shape[1], images.shape[2], images.shape[3], images.shape[4]))

    corr_avg = np.corrcoef(delta_gamma_flat, np.average(images_flat[:, :, :, :], axis=(1, 2, 3)))
    corr_red = np.corrcoef(delta_gamma_flat, np.average(images_flat[:, :, :, 0], axis=(1, 2))) - corr_avg
    corr_green = np.corrcoef(delta_gamma_flat, np.average(images_flat[:, :, :, 1], axis=(1, 2))) - corr_avg
    corr_blue = np.corrcoef(delta_gamma_flat, np.average(images_flat[:, :, :, 2], axis=(1, 2))) - corr_avg



if __name__ == "__main__":
    main()
