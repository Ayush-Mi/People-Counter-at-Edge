import torch
import argparse

def convert_to_onnx(cfgfile,weightfile):
    model = Darknet(cfgfile)
    model.load_weights(weightfile)
    model.eval()

    x = torch.randn(1,3,512,512)

    print("Converting to onxx model")
    output = torch.onnx.export(model,(x), "yolov4.onnx",opset_version = 11,verbose=False)
    print("sucessfully converted to yolov4.onxx")

    return True

def get_args():
    parser = argparse.ArgumentParser('Converting pytorch version of Yolov4 to Onnx.')
    parser.add_argument('-cfgfile', type=str, required=True,
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str, 
    	required=True,
        help='path of trained model.', dest='weightfile')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
	args = get_args()
	convert_to_onnx(args.configfile,args.weightfile)
