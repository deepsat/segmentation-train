from tensorflow.keras.optimizers import Adam
epochs = 100

tab=["fcn_8", "fcn_32", "fcn_8_vgg", "fcn_32_vgg", "fcn_8_resnet50", "fcn_32_resnet50", "fcn_8_mobilenet", "fcn_32_mobilenet", "pspnet", "vgg_pspnet", "resnet50_pspnet","unet_mini",
    "unet", "vgg_unet", "resnet50_unet", "mobilenet_unet","segnet","vgg_segnet","resnet50_segnet","mobilenet_segnet"]
from keras_segmentation.models.unet import resnet50_unet
model = resnet50_unet(n_classes=256,  input_height=256, input_width=256)

def depth_loss(y_true, y_pred):
  w1, w2, w3 = 1.0, 3.0, 0.1

  l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

  dy_true, dx_true = tf.image.image_gradients(y_true)
  dy_pred, dx_pred = tf.image.image_gradients(y_pred)
  l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)
  print(l_edges)

  l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, 1.0)) * 0.5, 0, 1)

  return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))

opt = Adam(lr=0.001, amsgrad=True)


model.compile(optimizer=opt, loss=depth_loss)

URLS = {
    'dataset-sample' : 'https://dl.dropboxusercontent.com/s/h8a8kev0rktf4kq/dataset-sample.tar.gz?dl=0',
    'dataset-medium' : 'https://dl.dropboxusercontent.com/s/r0dj9mhyv4bgbme/dataset-medium.tar.gz?dl=0',
}
LABELS = ['BUILDING', 'CLUTTER', 'VEGETATION', 'WATER', 'GROUND', 'CAR']

# Class to color (BGR)
LABELMAP = {
    0 : (255,   0, 255),
    1 : (75,   25, 230),
    2 : (180,  30, 145),
    3 : (75,  180,  60),
    4 : (48,  130, 245),
    5 : (255, 255, 255),
    6 : (200, 130,   0),
}

# Color (BGR) to class
INV_LABELMAP = {
    (255,   0, 255) : 0,
    (75,   25, 230) : 1,
    (180,  30, 145) : 2,
    (75,  180,  60) : 3,
    (48,  130, 245) : 4,
    (255, 255, 255) : 5,
    (200, 130,   0) : 6,
}

LABELMAP_RGB = { k: (v[2], v[1], v[0]) for k, v in LABELMAP.items() }

INV_LABELMAP_RGB = { v: k for k, v in LABELMAP_RGB.items() }

train_ids = [
    "1d4fbe33f3_F1BE1D4184INSPIRE",
    "1df70e7340_4413A67E91INSPIRE",
    "274518390f_AFAC6311B8OPENPIPELINE",
    "32760710b0_EF73EE9CCDOPENPIPELINE",
    "7008b80b00_FF24A4975DINSPIRE",
    "e2e401ba8b_CFF58D01D0OPENPIPELINE",
    "c644f91210_27E21B7F30OPENPIPELINE",
    "edc59d4824_FE5B96942BOPENPIPELINE",
    "b705d0cc9c_E5F5E0E316OPENPIPELINE",
    "ade6e4b261_147755FEAAOPENPIPELINE",
    "3bdbe137a1_E1B9B139DEOPENPIPELINE",
    "84830cff24_FE5B96942BOPENPIPELINE",
    "564d5fd4ea_F7D81C1243OPENPIPELINE",
    "a1af86939f_F1BE1D4184OPENPIPELINE",
    "571ed24019_7EF127EDCFOPENPIPELINE",
    "a0cee5daca_9ABAFDAA93OPENPIPELINE",
    "520947aa07_8FCB044F58OPENPIPELINE",
    "2ef883f08d_F317F9C1DFOPENPIPELINE",
    "f971256246_MIKEINSPIRE",
    "2ef3a4994a_0CCD105428INSPIRE",
    "56e9e81013_C988C95F03INSPIRE",
    "888432f840_80E7FD39EBINSPIRE",
    "63430fa268_B4DE0FB544INSPIRE",
    "130a76ebe1_68B40B480AOPENPIPELINE",
    "d02ce7cb10_6DC1FE1DDCOPENPIPELINE",
    "91ad290806_3CB2E8FC73INSPIRE",
    "11cdce7802_B6A62F8BE0INSPIRE",
    "6500c05298_B00063DE8EOPENPIPELINE",
    "803cd2c508_C988C95F03INSPIRE",
    "d5107a09cf_6ABE00F5A1INSPIRE",
    "3502e187b2_23071E4605OPENPIPELINE",
    "3452561694_E44D97430AOPENPIPELINE",
    "f9f43e5144_1DB9E6F68BINSPIRE",
    "2c36a93b10_793BC93268OPENPIPELINE",
    "53471726bc_B69D2F059FOPENPIPELINE",
    "afb793674b_4B44AF2928OPENPIPELINE",
    "807c0c243b_EA5BB57953OPENPIPELINE",
    "385393ca4b_E21EAB978AOPENPIPELINE",
    "6664b45691_D1F6B2028BOPENPIPELINE",
    "236da542ee_597D7FF2F9OPENPIPELINE",
    "7197260eb8_9549AC1A09INSPIRE",
    "7ed68b136e_C966B12B4EOPENPIPELINE",
    "1553627230_APIGENERATED",
    "ebffe540d0_7BA042D858OPENPIPELINE",
    "a4580732ce_2F98B8FC82INSPIRE",
    "c167ca6cb2_3CB2E8FC73INSPIRE",
    "e848b35eff_5EAE4DDF80INSPIRE",
    "d9161f7e18_C05BA1BC72OPENPIPELINE",
    "d45d74e584_2E8C142043OPENPIPELINE",
    "34fbf7c2bd_E8AD935CEDINSPIRE",
    "15efe45820_D95DF0B1F4INSPIRE",
    "2552eb56dd_2AABB46C86OPENPIPELINE",
    "628be3d244_A8CB55BF1FINSPIRE",
    "1553642501_APIGENERATED",
    "364d26fd40_9549AC1A09OPENPIPELINE",
    "1553541487_APIGENERATED",
    "ab4a9b813f_B75AF9044COPENPIPELINE",
    "748d0acb6d_18BE858545OPENPIPELINE",
    "686c48a300_9C340F2D92OPENPIPELINE",
    "5a5e4e491b_D7A795B2DEOPENPIPELINE",
    "a8789b3c97_1381767170OPENPIPELINE",
    "84410645db_8D20F02042OPENPIPELINE",
    "7c53bbf0da_EAFCA9B26AOPENPIPELINE",
    "9e7f0310a0_24E090DDB9INSPIRE",
    "b970fca868_883F63EBCCOPENPIPELINE",
    "d8786926c5_A6879692DAOPENPIPELINE",
    "d0dc53f9c7_9C194DD066INSPIRE",
    "2a8617d7d4_9464BAFE8AOPENPIPELINE",
    "f0747ed88d_E74C0DD8FDOPENPIPELINE",
    "c6d131e346_536DE05ED2OPENPIPELINE",
    "b61673f780_4413A67E91INSPIRE",
    "277f16713e_5E3246E306OPENPIPELINE",
    "7c59b1a217_B4DE0FB544INSPIRE",
    "74a2f5aaa9_B943F74EC9OPENPIPELINE",
    "1666d0369f_48FE7F729BOPENPIPELINE",
    "7c719dfcc0_310490364FINSPIRE",
    "f56b6b2232_2A62B67B52OPENPIPELINE",
    "c37dbfae2f_84B52814D2OPENPIPELINE",
    "c6890f580c_AFAC6311B8OPENPIPELINE",
    "399c1c010d_7A89E00BBDOPENPIPELINE",
    "87eeb1b9cc_B943F74EC9OPENPIPELINE",
    "7a14002b7b_B6E1859E4FINSPIRE",
    "5fa39d6378_DB9FF730D9OPENPIPELINE",
    "f4dd768188_NOLANOPENPIPELINE",
    "b771104de5_7E02A41EBEOPENPIPELINE",
    "981755057f_3BFBF39957OPENPIPELINE",
    "6958d7a8d5_9C194DD066OPENPIPELINE",
    "83d72f744d_48BF12F23COPENPIPELINE",
    "664e38b92b_6C7C9BE1D3INSPIRE",
    "1553541585_APIGENERATED",
    "c8eb574986_CC5FAE4CF9INSPIRE",
    "1d056881e8_29FEA32BC7INSPIRE",
    "fc5837dcf8_7CD52BE09EINSPIRE",
    "3a2200b6c0_2F98B8FC82OPENPIPELINE",
]

val_ids = [
    "ec09336a6f_06BA0AF311OPENPIPELINE",
    "679850f980_27920CBE78OPENPIPELINE",
    "c8a7031e5f_32156F5DC2INSPIRE",
    "6b82bcd67b_2EBB40A325OPENPIPELINE",
    "cc4b443c7d_A9CBEF2C97INSPIRE",
    "12c3372a95_7EF127EDCFINSPIRE",
    "941cb687d3_48FE7F729BINSPIRE",
    "42ab9f9e27_3CB2E8FC73INSPIRE",
    "264c36d368_C988C95F03INSPIRE",
    "954a8c814c_267994885AINSPIRE",
    "ea607f191d_582C2A2F47OPENPIPELINE",
    "600023a2df_F4A3C2E777INSPIRE",
    "57426ebe1e_84B52814D2OPENPIPELINE",
    "cd5a0d3ce4_2F98B8FC82INSPIRE",
    "3731e901b0_9464BAFE8AOPENPIPELINE",
    "f0c32df5a8_0406E6C238OPENPIPELINE",
    "1476907971_CHADGRISMOPENPIPELINE",
    "97c4dd388d_4C51642B86OPENPIPELINE",
    "f78c4e5748_3572E1D9BBOPENPIPELINE",
    "a11d963a7d_EF73EE9CCDOPENPIPELINE",
    "aef48b9aca_0226FDD487OPENPIPELINE",
    "9170479165_625EDFBAB6OPENPIPELINE",
    "3bb457cde8_D336A13367INSPIRE",
    "a1199a489f_6ABE00F5A1OPENPIPELINE",
    "137f4dfb89_C966B12B4EOPENPIPELINE",
    "551063e3c5_8FCB044F58INSPIRE",
    "37cf2e5706_74D898C7C3OPENPIPELINE",
    "74d7796531_EB81FE6E2BOPENPIPELINE",
    "46b27f92c2_06BA0AF311OPENPIPELINE",
    "32052d9b97_9ABAFDAA93OPENPIPELINE",
]

test_ids = [
    "12fa5e614f_53197F206FOPENPIPELINE",
    "feb7a50f10_JAREDINSPIRE",
    "c2e8370ca3_3340CAC7AEOPENPIPELINE",
    "55ca10d9f1_E8C8441957INSPIRE",
    "5ab849ec40_2F98B8FC82INSPIRE",
    "9254c82db0_9C194DD066OPENPIPELINE",
    "168ac179d9_31328BCCC4OPENPIPELINE",
    "6f93b9026b_F1BFB8B17DOPENPIPELINE",
    "8b0ac1fc28_6688905E16OPENPIPELINE",
    "1553539551_APIGENERATED",
    "7310356a1b_7EAE3AC26AOPENPIPELINE",
    "632de91030_9ABAFDAA93OPENPIPELINE",
    "2f7aabb6e5_0C2B5F6CABOPENPIPELINE",
    "18072ccb69_B2AE5C54EBOPENPIPELINE",
    "8710b98ea0_06E6522D6DINSPIRE",
    "fb74c54103_6ABE00F5A1INSPIRE",
    "25f1c24f30_EB81FE6E2BOPENPIPELINE",
    "39e77bedd0_729FB913CDOPENPIPELINE",
    "e87da4ebdb_29FEA32BC7INSPIRE",
    "546f85625a_39E021DC32INSPIRE",
    "e1d3e6f6ba_B4DE0FB544INSPIRE",
    "eee7d707d4_6DC1FE1DDCOPENPIPELINE",
    "3ff76e84d5_0DD77DFCD7OPENPIPELINE",
    "a0a6f46099_F93BAE5403OPENPIPELINE",
    "420d6b69b8_84B52814D2OPENPIPELINE",
    "d06b2c67d2_2A62B67B52OPENPIPELINE",
    "107f24d6e9_F1BE1D4184INSPIRE",
    "36d5956a21_8F4CE60B77OPENPIPELINE",
    "1726eb08ef_60693DB04DINSPIRE",
    "dabec5e872_E8AD935CEDINSPIRE",
]

import cv2
import os
import numpy as np

size   = 256
stride = 256

def image2tile(prefix, scene, dataset, orthofile, elevafile, labelfile, windowx=size, windowy=size, stridex=stride, stridey=stride):

    ortho = cv2.imread(orthofile)
    label = cv2.imread(elevafile, -1)

    # Not using elevation in the sample - but useful to incorporate it ;)
    eleva = cv2.imread(elevafile, -1)

    assert(ortho.shape[0] == label.shape[0])
    assert(ortho.shape[1] == label.shape[1])

    shape = ortho.shape

    xsize = shape[1]
    ysize = shape[0]
    print(f"converting {dataset} image {orthofile} {xsize}x{ysize} to chips ...")

    counter = 0

    for xi in range(0, shape[1] - windowx, stridex):
        for yi in range(0, shape[0] - windowy, stridey):

            orthochip = ortho[yi:yi+windowy, xi:xi+windowx, :]
            labelchip = label[yi:yi+windowy, xi:xi+windowx]

            orthochip_filename = os.path.join(prefix, 'image-chips', scene + '-' + str(counter).zfill(6) + '.png')
            labelchip_filename = os.path.join(prefix, 'label-chips', scene + '-' + str(counter).zfill(6) + '.png')

            with open(f"{prefix}/{dataset}", mode='a') as fd:
                fd.write(scene + '-' + str(counter).zfill(6) + '.png\n')

            cv2.imwrite(orthochip_filename, orthochip)
            cv2.imwrite(labelchip_filename, labelchip)
            counter += 1


def get_split(scene):
    if scene in train_ids:
        return "train.txt"
    if scene in val_ids:
        return 'valid.txt'
    if scene in test_ids:
        return 'test.txt'

def run(prefix):

    open(prefix + '/train.txt', mode='w').close()
    open(prefix + '/valid.txt', mode='w').close()
    open(prefix + '/test.txt', mode='w').close()

    if not os.path.exists( os.path.join(prefix, 'image-chips') ):
        os.mkdir(os.path.join(prefix, 'image-chips'))

    if not os.path.exists( os.path.join(prefix, 'label-chips') ):
        os.mkdir(os.path.join(prefix, 'label-chips'))


    lines = [ line for line in open(f'{prefix}/index.csv') ]
    num_images = len(lines) - 1
    print(f"converting {num_images} images to chips - this may take a few minutes but only needs to be done once.")

    for lineno, line in enumerate(lines):

        line = line.strip().split(' ')
        scene = line[1]
        dataset = get_split(scene)

        if dataset == 'test.txt':
            print(f"not converting test image {scene} to chips, it will be used for inference.")
            continue

        orthofile = os.path.join(prefix, 'images',     scene + '-ortho.tif')
        elevafile = os.path.join(prefix, 'elevations', scene + '-elev.tif')
        labelfile = os.path.join(prefix, 'labels',     scene + '-label.png')

        if os.path.exists(orthofile) and os.path.exists(labelfile):
            image2tile(prefix, scene, dataset, orthofile, elevafile, labelfile)

def download_dataset(dataset):
    """ Download a dataset, extract it and create the tiles """

    if dataset not in URLS:
        print(f"unknown dataset {dataset}")
        sys.exit(0)

    filename = f'{dataset}.tar.gz'
    url = URLS[dataset]

    if not os.path.exists(filename):
        print(f'downloading dataset "{dataset}"')
        os.system(f'curl "{url}" -o {filename}')
    else:
        print(f'zipfile "{filename}" already exists, remove it if you want to re-download.')

    if not os.path.exists(dataset):
        print(f'extracting "{filename}"')
        os.system(f'tar -xvf {filename}')
    else:
        print(f'folder "{dataset}" already exists, remove it if you want to re-create.')

    image_chips = f'{dataset}/image-chips'
    label_chips = f'{dataset}/label-chips'
    if not os.path.exists(image_chips) and not os.path.exists(label_chips):
        print("creating chips")
        run(dataset)
    else:
        print(f'chip folders "{image_chips}" and "{label_chips}" already exist, remove them to recreate chips.')
        
import sys
import os
download_dataset('dataset-medium')

dataset = "dataset-sample/"
model.train(
    train_images =  "dataset-medium/image-chips/",
    train_annotations = "dataset-medium/label-chips/",
    checkpoints_path = "resnet50_unet", epochs=epochs
)
model.save("elevation.h5")
