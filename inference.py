# Import libraries
import torch, yaml, os, pickle, timm, argparse
from utils import get_state_dict, get_preds, visualize, grad_cam

def run(args):
    
    """
    
    This function runs the infernce script based on the arguments.
    
    Parameter:
    
        args - parsed arguments.
        
    Output:
    
        train process.
    
    """
    
    # Get train arguments 
    argstr = yaml.dump(args.__dict__, default_flow_style = False)
    print(f"\nTraining Arguments:\n\n{argstr}")
    
    # Create a directory to save inference results
    os.makedirs(args.save_path, exist_ok = True)
    
    # Get the saved test dataloader
    test_dl = torch.load(f"{args.dls_dir}/test_dl")
    # Get the saved class names file
    with open(f"{args.dls_dir}/cls_names.pkl", "rb") as f: cls_names = pickle.load(f)
    print(f"Dataloader and class names are successfully loaded!")
    print(f"There are {len(test_dl)} batches and {len(cls_names)} classes in the test dataloader!")

    # Get the model to be used during inference
    model = timm.create_model(args.model_name, num_classes = len(cls_names)); model.to(args.device)
    # Load the parameters of the trained model
    print("\nLoading the state dictionary...")
    state_dict = f"{args.save_model_path}/med_best_model.pth"
    model.load_state_dict(torch.load(state_dict, map_location = "cpu"), strict = True)
    print(f"The {args.model_name} state dictionary is successfully loaded!\n")
    # Get images, predictions, and labels
    all_ims, all_preds, all_gts = get_preds(model, test_dl, args.device)
    # Visualize the inference results
    visualize(all_ims, all_preds, all_gts, num_ims = 10, rows = 2, cls_names = cls_names, save_path = args.save_path, save_name = args.dataset_name)
    # GradCAM of the inference results
    grad_cam(model, all_ims, num_ims = 10, rows = 2, save_path = args.save_path, save_name = args.dataset_name)
    
if __name__ == "__main__":
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description = "Inference Process Arguments")
    
    # Add arguments to the parser
    parser.add_argument("-is", "--inp_im_size", type = tuple, default = (224, 224), help = "Input image size")
    parser.add_argument("-dn", "--dataset_name", type = str, default = "cars", help = "Dataset name for training")
    parser.add_argument("-mn", "--model_name", type = str, default = "rexnet_150", help = "Model name for backbone")
    parser.add_argument("-d", "--device", type = str, default = "cuda:3", help = "GPU device name")
    parser.add_argument("-sm", "--save_model_path", type = str, default = "saved_models", help = "Path to the directory to save a trained model")
    parser.add_argument("-sp", "--save_path", type = str, default = "results", help = "Path to dir to save inference results")
    parser.add_argument("-dl", "--dls_dir", type = str, default = "saved_dls", help = "Path to dir to save dataloaders")
    
    # Parse the added arguments
    args = parser.parse_args() 
    
    # Run the script with the parsed arguments
    run(args)
