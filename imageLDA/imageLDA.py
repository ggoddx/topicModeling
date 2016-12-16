from PIL import Image
from sklearn.decomposition import LatentDirichletAllocation

import cPickle, csv, getSysArgs
import matplotlib.pyplot as plt
import numpy as np


## Unpickles files and returns a dictionary
#  function code credit: https://www.cs.toronto.edu/~kriz/cifar.html
#
#  @param fPath string
#   filepath (including filename) for pickle file
#
#  @return dict dictionary
#   dictionary of data within pickle file
def unpickle(fPath):
    ## Open file
    fIn = open(fPath, 'rb')

    ## Dictionary in pickle file
    dict = cPickle.load(fIn)

    fIn.close()

    return dict


def main():
    ## Pickle file of images and categories
    [fImages,
     fCategs,
     n_topics] = getSysArgs.usage(['imageLDA.py',
                                   '<image_pickle_file_path>',
                                   '<image_categories_file_path>',
                                   '<number_of_topics>'])[1:]

    n_topics = int(n_topics)

    ## Batch of images
    images = unpickle(fImages)

    ## Categories for images
    categs = unpickle(fCategs)

    ## LDA model
    lda = LatentDirichletAllocation(n_topics = n_topics, max_iter = 5,
                                    learning_method = 'online')

    ## Image vectors
    imVecs = images['data']

    lda.fit(imVecs)

    ## Each image's topic distribution
    im2topic = lda.transform(imVecs)

    ## File to write topic distributions
#    tdCSV = csv.writer(open('im2topic.csv', 'wb', buffering = 0))

#    tdCSV.writerows(im2topic)

    ## Range of topic numbers
    topicRange = range(n_topics)

    for i in topicRange:
        ## Top 10 images indices for topic
        top10indices = im2topic[:, i].argsort()[::-1][:10]

        for j in top10indices:
            ## Top image vector
            topImg = images['data'][j]

            ## Top vector arranged as red, green, and blue pixels
            topImgRGB = {'R': topImg[:1024].reshape(32, 32, 1),
                         'G': topImg[1024:2048].reshape(32, 32, 1),
                         'B': topImg[2048:].reshape(32, 32, 1)}

            topImg = topImgRGB['R']
            topImg = np.append(topImg, topImgRGB['G'], axis = 2)
            topImg = np.append(topImg, topImgRGB['B'], axis = 2)

            ## To save image as a file
            im = Image.fromarray(topImg)

            im.save('./' +  str(n_topics) + 'topics/topic' + str(i) + '/' + images['filenames'][j])

    return

if __name__ == '__main__':
    main()
