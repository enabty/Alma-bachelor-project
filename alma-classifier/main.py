from alma_classifier import pipeline_tensorflow
from alma_classifier.image_processing.manual_sorting import predict_fits
from keras.models import load_model

def main(): 


    # model = pipeline_tensorflow.pippeline_tensorflow(linnear_aug=True, augmentation_factor=10)

    # model.save('test_model')

    # model = load_model('test_model')




    # images_of_interest = predict_fits('C:/ChalmersWorkspaces/KandidatArbete/raw_data/neg', model)

    # images_of_interest = predict_fits('alma-classifier/data/fits/pos', model)





if __name__ == '__main__': main()
