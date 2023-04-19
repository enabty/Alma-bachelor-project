from image_processing.pre_processing import pre_processing
from image_processing.image_augmentation import neg_image_augmentation, pos_image_augmentation
from pathlib import Path


def generate_training_data(gen_data=bool):
    if gen_data == True:

        print('Generating training data')

        root_path = str(Path(__file__).parents[2])
        pos_data = pre_processing(
            root_path + '\\alma-classifier\\data\\fits\\pos\\')
        pos_data = pos_image_augmentation(pos_data)

        neg_data = pre_processing(
            root_path + '\\alma-classifier\\data\\fits\\neg\\')
        neg_data = neg_image_augmentation(neg_data)

        # ----------------------Save data to .npy file----------------------#
        # np.save('pos_train.npy', pos_data, allow_pickle=True)
        # np.save('neg_train.npy', neg_data, allow_pickle=True)

        print('Data generated successfully')
        return neg_data


__name__ == '__main__' and generate_training_data(True)
