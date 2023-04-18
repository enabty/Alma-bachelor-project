from pre_processing import pre_processing
from image_augmentation import neg_image_augmentation, pos_image_augmentation


def generate_training_data(gen_data=bool):
    if gen_data == True:

        print('Generating training data')

        pos_data = pre_processing(
            'C:\ChalmersWorkspaces\KandidatArbete\Alma-bachelor-project\data\pos_fits')
        pos_data = pos_image_augmentation(pos_data)

        neg_data = pre_processing(
            'C:\ChalmersWorkspaces\KandidatArbete\Alma-bachelor-project\data\neg_fits')
        neg_data = neg_image_augmentation(neg_data)

        # ----------------------Save data to .npy file----------------------#
        # np.save('pos_train.npy', pos_data, allow_pickle=True)
        # np.save('neg_train.npy', neg_data, allow_pickle=True)

        print('Data generated successfully')
        return neg_data


__name__ == '__main__' and generate_training_data(True)
