import os
import dgl
import torch
import torch.optim as optim
from tqdm import tqdm
from datasets.kitti import KITTI
from datasets.utils import euler_to_rotation_torch
from build_model import build_loop_model
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import pickle
import json
from tensorboard import program


torch.manual_seed(2023)


def val_epoch(model, val_loader, criterion, args):
    epoch_loss = 0
    with tqdm(val_loader, unit="batch", ncols=70) as tepoch:
        # for images, gt, g_gt in tepoch:
        for images,gt in tepoch:
            tepoch.set_description(f"Validating ")
            # for batch_idx, (images, odom) in enumerate(train_loader):
            if torch.cuda.is_available():
                images,gt = images.cuda(), gt.cuda()
            src = list(range(images.shape[0]))
            dst = list(range(1, images.shape[0] + 1))
            g = dgl.graph((src, dst)).to('cuda')

            # predict pose
            estimated_pose = model(images.float(), g)

            # compute loss
            loss = compute_loss(estimated_pose, gt, criterion, args)

            epoch_loss += loss.item()
            tepoch.set_postfix(val_loss=loss.item())

    return epoch_loss / len(val_loader)


import functools

def train_epoch(model, train_loader, criterion, optimizer, epoch, tensorboard_writer, args):
    epoch_loss = 0
    iter = (epoch - 1) * len(train_loader) + 1
    mean_angles = torch.Tensor([1.7061e-5, 9.5582e-4, -5.5258e-5]).cuda()
    std_angles = torch.Tensor([2.8256e-3, 1.7771e-2, 3.2326e-3]).cuda()
    mean_t = torch.Tensor([-8.6736e-5, -1.6038e-2, 9.0033e-1]).cuda()
    std_t = torch.Tensor([2.5584e-2, 1.8545e-2, 3.0352e-1]).cuda()

    with tqdm(train_loader, unit="batch", ncols=70) as tepoch:
        # for images, gt, g_gt in tepoch:
        for images, gt in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            # for batch_idx, (images, odom) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, gt = images.cuda(), gt.cuda()

            # predict pose
            # estimated_pose, g_pose = model(images.float())
            src = list(range(images.shape[0]))
            dst = list(range(1, images.shape[0] + 1))
            g = dgl.graph((src, dst)).to('cuda')
            refine_pose= model(images.float(), g)


            
            loss = compute_loss(refine_pose, gt, criterion, args) 
            # compute loss
            # + compute_loss(g_pose, g_gt.cuda(), criterion, args)

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

            # log tensorboard
            # tensorboard_writer.add_scalar('training_loss', loss.item(), iter)
            iter += 1
    return epoch_loss / len(train_loader)  
  

def train(model, train_loader, val_loader, criterion, optimizer, tensorboard_writer, args):
    checkpoint_path = args["checkpoint_path"]
    epochs = args["epoch"]
    best_val = args["best_val"]
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(args["epoch_init"], epochs):
        # training for one epoch
        model.train()
        # model.sp.eval()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, tensorboard_writer, args)
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "best_val": best_val,
        }

        # validate model
        if val_loader and not epoch%1:
            with torch.no_grad():
                model.eval()
                val_loss = val_epoch(model, val_loader, criterion, args)

            print(f"Epoch: {epoch} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} \n")

            # save best mode
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "best_val": best_val,
            }
            if val_loss < best_val:
                print(f"Saving new best model -- loss decreased from {best_val:.6f} to {val_loss:.6f} \n")
                best_val = val_loss
                state["best_val"] = best_val
                torch.save(state, os.path.join(checkpoint_path, "checkpoint_best.pth"))

            # log validation loss in TensorBoard
            tensorboard_writer.add_scalar("val_loss", val_loss, epoch)

        # save checkpoint every 20 epochs
        if not epoch%20:
            torch.save(state, os.path.join(checkpoint_path, "checkpoint_e{}.pth".format(epoch))) 
        # save last checkpoint
        torch.save(state, os.path.join(checkpoint_path, "checkpoint_last.pth"))  

        # log loss in TensorBoard
        tensorboard_writer.add_scalar("train_loss", train_loss, epoch)
    return


def get_optimizer(params, args):
    method = args["optimizer"]

    # initialize the optimizer
    if method == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, params), lr=args["lr"])
    elif method == "SGD":
        optimizer = optim.SGD(params, lr=args["lr"],
                              momentum=args["momentum"],
                              weight_decay=args["weight_decay"])
    elif method == "RAdam":
        optimizer = optim.RAdam(params, lr=args["lr"])
    elif method == "Adagrad":
        optimizer = optim.Adagrad(params, lr=args["lr"],
                                  weight_decay=args["weight_decay"])

    # load checkpoint
    if args["checkpoint"] is not None:
        checkpoint = torch.load(os.path.join(args["checkpoint_path"], args["checkpoint"]))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return optimizer


def compute_loss(y_hat, y, criterion, args):
    if args["weighted_loss"] == None:
        loss = criterion(y_hat, y.float())
    else:
        y = torch.reshape(y, (y.shape[0], args["window_size"]-1, 6))
        gt_angles = y[:, :, :3].flatten()
        gt_translation = y[:, :, 3:].flatten()

        # predict pose
        y_hat = torch.reshape(y_hat, (y_hat.shape[0], args["window_size"]-1, 6))
        estimated_angles = y_hat[:, :, :3].flatten()
        estimated_translation = y_hat[:, :, 3:].flatten()

        # compute custom loss
        k = args["weighted_loss"]
        loss_angles = k * criterion(estimated_angles, gt_angles.float())
        loss_translation = criterion(estimated_translation, gt_translation.float())
        loss =  loss_angles + loss_translation   
    return loss


if __name__ == "__main__":

    # set hyperparameters and configuration
    args = {
        "data_dir": "data",
        "bsize": 4,  # batch size
        "val_split": 0.1,  # percentage to use as validation data
        "window_size": 2,  # number of frames in window
        "overlap": 1,  # number of frames overlapped between windows
        "optimizer": "Adam",  # optimizer [Adam, SGD, Adagrad, RAdam]
        "lr": 1e-5,  # learning rate
        "momentum": 0.9,  # SGD momentum
        "weight_decay": 1e-4,  # SGD momentum
        "epoch": 300,  # train iters each timestep
    	"weighted_loss": None,  # float to weight angles in loss function
    	"pretrained_ViT": False,  # load weights from pre-trained ViT
        "checkpoint_path": "checkpoints/Exp14",  # path to save checkpoint
        "checkpoint": "checkpoint_best.pth",  # checkpoint
        # "checkpoint": None,  # checkpoint
    }

    # tiny  - patch_size=16, embed_dim=192, depth=12, num_heads=3
    # small - patch_size=16, embed_dim=384, depth=12, num_heads=6
    # base  - patch_size=16, embed_dim=768, depth=12, num_heads=12
    model_params = {
        "dim": 384,
        "image_size": (192, 640),  #(192, 640),
        "patch_size": 16,
        "attention_type": 'divided_space_time',  # ['divided_space_time', 'space_only','joint_space_time', 'time_only']
        "num_frames": args["window_size"],
        "num_classes": 6 * (args["window_size"] - 1),  # 6 DoF for each frame
        "depth": 12,
        "heads": 6,
        "dim_head": 64,
        "attn_dropout": 0.1,
        "ff_dropout": 0.1,
        "time_only": False,
    }
    args["model_params"] = model_params

    # create checkpoints folder
    if not os.path.exists(args["checkpoint_path"]):
        os.makedirs(args["checkpoint_path"])

    with open(os.path.join(args["checkpoint_path"], 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)
    with open(os.path.join(args["checkpoint_path"], 'args.txt'), 'w') as f:
    	f.write(json.dumps(args))

    # tensorboard writer
    TensorBoardWriter = SummaryWriter(log_dir=args["checkpoint_path"])
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir=checkpoints/Exp14', '--bind_all', '--port=6006']) 
    url = tb.launch()
    print(f"TensorBoard URL ：{url}")


    # preprocessing operation
    preprocess = transforms.Compose([
        transforms.Resize((model_params["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.34721234, 0.36705238, 0.36066107],
            std=[0.30737526, 0.31515116, 0.32020183]),
    ])


    # train and val dataloader
    print("Using CUDA: ", torch.cuda.is_available())
    print("Loading data...")
    dataset = KITTI(window_size=args["window_size"], overlap=args["overlap"], transform=preprocess, train=True)
    nb_val = round(args["val_split"] * len(dataset))

    train_data, val_data = random_split(dataset, [len(dataset) - nb_val, nb_val]) #generator=torch.Generator().manual_seed(2))
    
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args["bsize"],
                                               shuffle=False,
                                               )
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=4,
                                             shuffle=False,
                                             )

    # build and load model
    print("Building model...")
    model, args = build_loop_model(args, model_params)

    n = sum([param.nelement() for param in model.parameters()])
    print(n)

    # loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = get_optimizer(model.parameters(), args)

    # train network
    print(20*"--" +  " Training " + 20*"--")
    train(model, train_loader, val_loader, criterion, optimizer, TensorBoardWriter, args)
