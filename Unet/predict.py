import glob
import numpy as np
import torch
import cv2
from model.unet_model import UNet

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load model
    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    net.eval()
    # Load img path
    tests_path = glob.glob('data/test/*.png')
    # Start predict
    for test_path in tests_path:
        # save dir
        save_res_path = test_path.split('.')[0] + '_res.png'
        # read img
        img = cv2.imread(test_path)
        # transfer as self definded dataset
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)

        # predict
        pred = net(img_tensor)
        # extract results
        pred = np.array(pred.data.cpu()[0])[0]
        print(test_path)
        # handle results
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # save img
        cv2.imwrite(save_res_path, pred)