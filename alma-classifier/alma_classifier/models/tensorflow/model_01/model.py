
# from astropy.io import fits
# import glob


def generate_training_data(gen_data=bool):
    if gen_data == True:

        print('Generating training data')

        # pos_data = pre_processing(
        #     'C:\ChalmersWorkspaces\KandidatArbete\Alma-bachelor-project\data\pos_fits')

        # pos_data = pos_image_augmentation(pos_data)

        # np.save('pos_train.npy', pos_data, allow_pickle=True)

        neg_data = pre_processing(
            'C:\ChalmersWorkspaces\KandidatArbete\Alma-bachelor-project\data\neg_fits')

        print(len(neg_data))

        neg_data = neg_image_augmentation(neg_data)

        # np.save('neg_train.npy', neg_data, allow_pickle=True)

        # print(len(neg_data), len(pos_data))

        print('Data generated successfully')

        return neg_data


__name__ == '__main__' and generate_training_data(True)
