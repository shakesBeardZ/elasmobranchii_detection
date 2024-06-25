from ultralytics import YOLO
import argparse
import os

def main(args):
    print(args)
    # Initialize model
    model = YOLO(args.weights)
    # Generate GPU device list
    gpu_devices = list(range(args.gpus))  # This creates a list of GPUs from 0 to args.gpus-1

    # Training
    model.train(data=args.data, epochs=args.epochs, batch=args.batch_size, imgsz=args.imgsz, device=gpu_devices, workers=args.workers)

    # Save final model
    model.save(os.path.join(args.save_dir, 'final_model.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLOv8 Model")
    parser.add_argument('--weights', type=str, default='yolov8x.pt', help='initial weights path')
    parser.add_argument('--data', type=str, default='/home/yahiab/rays_detection/data2/data.yaml', help='dataset configuration file')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--imgsz', type=int, default=2048, help='image size')
    parser.add_argument('--save_dir', type=str, default='./runs', help='directory to save models')
    parser.add_argument('--workers', type=int, default=6, help='number of workers for data loading')
    parser.add_argument('--gpus', type=int, default=2, help='number of GPUs to use')


    args = parser.parse_args()
    main(args)