import glob
import numpy as np
from matplotlib import pyplot as plt
from image_augmentation import neg_image_augmentation, pos_image_augmentation
from pre_processing import pre_processing
from astropy.visualization import ZScaleInterval
zscale = ZScaleInterval(contrast=0.25, nsamples=1)


def manual_sorting():
    pos_data = pre_processing(
        'C:/ChalmersWorkspaces/KandidatArbete/Alma-bachelor-project/data/fits/pos')
    print(len(pos_data))
    pos_data = pos_image_augmentation(pos_data)



    for i in range(0, len(pos_data)):
        plt.figure(frameon=False)
        plt.imshow(zscale(pos_data[i]), origin='lower',
                   cmap='CMRmap_r', aspect='auto')
        plt.savefig(
            'alma-classifier/alma_classifier/image_processing/temp/pos_{}.png'.format(i))
        plt.close()


def sort_pos_data():
    file_names = glob.glob('alma-classifier/alma_classifier/image_processing/temp/*.png')
    print(len(file_names))
    index_to_keep = []

    for file in file_names:
        if len(file) == 63:
            index_to_keep.append(int(file[-5:-4]))
        elif len(file) == 64:
            index_to_keep.append(int(file[-6:-4]))
        elif len(file) == 65:
            index_to_keep.append(int(file[-7:-4]))
    
    pos_data = pre_processing(
            'C:/ChalmersWorkspaces/KandidatArbete/Alma-bachelor-project/data/fits/pos')
    pos_data = pos_image_augmentation(pos_data)

    # Sort the data by index of index_to_keep

    for i in range(0, len(pos_data)):
        if i in index_to_keep:
            np.save('data/pos_dataset/pos_dataset.npy',
                    pos_data[i], allow_pickle=True)
    return index_to_keep


# def manual_sorting():
#     pos_data = pre_processing(
#         'C:/ChalmersWorkspaces/KandidatArbete/Alma-bachelor-project/data/fits/pos')
#     pos_data = pos_image_augmentation(pos_data)


#     save_fits_to_png(pos_data)

#     # Sort the data by hand via keyboard input
#     indexes = []
#     for i in range(0, len(pos_data)):
#         plt.imshow(pos_data[i], cmap='CMRmap_r')
#         print(f'Index: {i}')
#         print('Keep? (y/n)')
#         plt.show()
#         plt.close()
#         if input() == 'y':
#             indexes.append(i)
#         else:
#             pos_data[i].remove()

#     np.save('C:\ChalmersWorkspaces\KandidatArbete\Alma-bachelor-project\data\pos_dataset\pos_dataset.npy', pos_data, allow_pickle=True)
__name__ == '__main__' and sort_pos_data()
