import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.limitcycle import make_limitcycle_dataset
from utils.dataset import DynamicalSystemDataset
from PhaseReductionNet import Encoder, Decoder, LatentSteper

LOW_DIM_LIMIT_CYCLE = ['SL','VP','HH','FHN','FHNR']


def argsWrite(p,log_file):
    f = open(log_file, 'a')
    f.write('-----------Prameter-----------\n')
    args = [(i, getattr(p, i)) for i in dir(p) if not '_' in i[0]]
    for i, j in args:
        f.write('{0}:{1}\n'.format(i, j))
    f.write('------------------------------\n\n')


def to_polar(x):
    theta = torch.atan2(x[:, 1], x[:, 0])
    return theta


def main():
    """
    This code is the source code associated with the paper "Phase autoencoder for limit-cycle oscillators".
    link: https://arxiv.org/abs/2403.06992

    The code requires limitcycle orbits(limit_cycle_**.npy) and phase sensitive functions(phase_response_function_**.npy) ,
    but the method itself does not require these data.
    These data is used for generation data and evaluation.
    """
    # Get arguments
    parser = argparse.ArgumentParser()
    # Parameter (experimental management)
    parser.add_argument('--ex_name', type=str, default='ex')  # experiment ID
    # Parameter (limitcycle)
    parser.add_argument('--lc_name', type=str, default='SL')  # limitcycle(name),'SL','VP','HH','FHN3','FHNR'
    parser.add_argument('--dt', type=float, default=0.001)  # Computation time step width for dynamical systems.
    parser.add_argument('--noise_rate', type=float, default=0.5)  # Noise size
    parser.add_argument('--num_rotation', type=int, default=3)  # Number of limit cycle turns
    parser.add_argument('--data_interval', type=int, default=5)  # Parameters for how finely the data is taken. 
    # Parameter for thinning the data as the training data is huge when dt is small. 

    # モデル、学習に関するパラメータ
    parser.add_argument('--step_interval', type=int, default=10)
    parser.add_argument('--step_num', type=int, default=20)
    parser.add_argument('--latent_dim', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--epoch_size', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--w_step', type=float, default=0.1)
    parser.add_argument('--w_z1', type=float, default=2.0)
    parser.add_argument('--sw', type=float, default=1.0)
    # モード
    parser.add_argument('--check_data', action='store_true')
    parser.add_argument('--train_traj_num', type=int, default=-1)
    parser.add_argument('--noise_level', type=float, default=0.0)

    args = parser.parse_args()
    ex_name = args.ex_name
    lc_name = args.lc_name
    dt = args.dt
    noise_rate = args.noise_rate
    num_rotation = args.num_rotation
    data_interval = args.data_interval
    step_interval = args.step_interval
    step_num = args.step_num
    latent_dim = args.latent_dim
    hidden_dim = args.hidden_dim
    epoch_size = args.epoch_size
    batch_size = args.batch_size
    lr = args.lr
    w_step = args.w_step
    w_z1 = args.w_z1
    sw = args.sw
    sw1 = 1.0
    sw2 = 1.0
    only_check_data = args.check_data
    train_traj_num = args.train_traj_num
    noise_level = args.noise_level

    # ログファイルの作成と結果ディレクトリの作成
    if not os.path.exists(f'./out'):
        os.mkdir(f'./out')
    if not os.path.exists(f'./out/{ex_name}'):
        os.mkdir(f'./out/{ex_name}')
    log_file = f'./out/{ex_name}/log.txt'
    f = open(log_file, 'w')
    f.write('Logging Start.\n\n')
    f.close()
    argsWrite(args, log_file)
    lc_X0 = np.load(f'./data/limit_cycle_{lc_name}.npy')
    if lc_name in LOW_DIM_LIMIT_CYCLE:
        lc_prf = np.load(f'./data/phase_response_function_{lc_name}.npy')
    d = lc_X0.shape[1]
    lc_step = lc_X0.shape[0]
    if lc_name in LOW_DIM_LIMIT_CYCLE:
        mean_X0 = lc_X0.mean(axis=0)
        std_X0 = lc_X0.std(axis=0)
    else:
        #mean_X0 = np.zeros(d)
        #std_X0 = np.ones(d)
        mean_X0 = lc_X0.mean(axis=0)
        std_X0 = lc_X0.std(axis=0)
        for i in range(len(std_X0)):
            std_X0[i] = 0.5 #np.max([std_X0[i],0.1])
    print('Limit Cycle Dimension:', d)
    print('Number of Initial X:', lc_step)
    if dt != (-1):
        print('Period Time:', lc_step*dt)
    else:
        print('Period Time:', lc_step*0.001)
    if lc_name in LOW_DIM_LIMIT_CYCLE:
        X0 = []
        for n in range(1):
            _X0 = lc_X0.copy()
            for i in range(lc_X0.shape[1]):
                _X0[:, i] += np.random.randn(lc_step)*noise_rate*std_X0[i]
            X0.append(_X0)
        X0 = np.concatenate(X0, axis=0)
        print(X0.shape)
        rate = int(len(X0)/(train_traj_num/0.95))
        if rate >= 2:
            X0 = X0[::rate, :]
        print(X0.shape)
        data = make_limitcycle_dataset(model_nm=lc_name,
                                    X0=X0,
                                    num_rotation=num_rotation,
                                    dt=dt,
                                    data_interval=data_interval
                                    )
        
    else:
        pass
    
    print("data shape:", data.shape)
    data += np.random.randn(data.shape[0],data.shape[1],data.shape[2])*noise_level
    if only_check_data:
        #初期値の可視化
        #plt.plot(X0)
        #plt.plot(lc_X0)
        if data.shape[2]==2:
            #plt.plot(lc_X0[:, 0], lc_X0[:, 1])
            #plt.scatter(data[:, 0, 0], data[:, 0, 1], s=1)
            
            plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
            plt.rcParams["mathtext.fontset"] = "stix"
            plt.rcParams["font.size"] = 30 
            plt.figure(figsize=(8,6))
            for i in range(5):
                plt.plot(data[i,:, 0], data[i,:, 1])
            plt.xlabel(r'$x_1$')
            plt.ylabel(r'$x_2$')
            plt.savefig(f'./out/{ex_name}/data.pdf', bbox_inches='tight')
            plt.show()
    for i in range(lc_X0.shape[1]):
        data[:,:,i] -= mean_X0[i]
        data[:,:,i] /= std_X0[i]
    print(data.shape)  # 軌道数×ステップ数×システムの次元
    num_traj = data.shape[0]
    # 学習をせずにデータをチェックするだけで終わり
    if only_check_data:
        return

    train_traj = np.random.choice(num_traj, int(num_traj*0.95),
                                  replace=False)
    val_traj = list(set(list(range(num_traj)))-set(train_traj))
    if train_traj_num > 0 and train_traj_num < len(train_traj):
        train_traj = np.random.choice(train_traj, train_traj_num, replace=False)
    train_data = data[train_traj]
    val_data = data[val_traj]
    print(train_data.shape, val_data.shape)

    train_dataset = DynamicalSystemDataset(train_data,
                                           step_num=step_num,
                                           step_interval=step_interval)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)

    device = 'cuda'
    input_dim = lc_X0.shape[1]
    enc = Encoder(input_dim=input_dim, output_dim=latent_dim,
                  hidden_dim=hidden_dim)
    step = LatentSteper(zd=latent_dim-2)
    dec = Decoder(input_dim=latent_dim, output_dim=input_dim,
                  hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(list(enc.parameters())
                                 + list(dec.parameters())
                                 + list(step.parameters()),
                                 lr=lr)
    enc.to(device)
    step.to(device)
    dec.to(device)

    #dt_now = datetime.datetime.now()
    writer = SummaryWriter()

    print('Training Start')
    for e in range(epoch_size):
        loss_vec = []
        enc.train()
        step.train()
        dec.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            xs = torch.Tensor(batch).to(device, dtype=torch.float)
            x = xs[:, 0, :]
            z = enc(x)
            y = dec(z)
            loss_recon = torch.nn.MSELoss()(x, y)
            for step_i in range(1, step_num):
                if step_i == 1:
                    step_z = step(z)
                else:
                    step_z = step(step_z)
                step_x = xs[:, step_i, :]
                enc_z = enc(step_x)
                if step_i == 1:
                    loss_step_phase = torch.nn.MSELoss()(step_z[:,:2], enc_z[:,:2])
                    loss_step_r = torch.nn.MSELoss()(step_z[:,2:], enc_z[:,2:])
                else:
                    mw = np.power(step_i, sw)
                    loss_step_phase += torch.nn.MSELoss()(step_z[:,:2], enc_z[:,:2])/mw
                    loss_step_r += torch.nn.MSELoss()(step_z[:,2:], enc_z[:,2:])/mw
            loss_z1 = torch.norm(torch.mean(z[:, :2], dim=0))
            loss_z2 = torch.norm(z[:, 2:])

            #loss = loss_recon/d + loss_step * w_step + loss_z1 * w_z1
            loss = loss_recon + loss_step_phase * w_step * sw2 \
                + loss_step_r * w_step + loss_z1 * w_z1 * sw1
            loss.backward()
            optimizer.step()
            loss_vec.append(np.array([loss.item(),
                                      loss_recon.item(),
                                      loss_step_phase.item(),
                                      loss_step_r.item(),
                                      loss_z1.item(),
                                      loss_z2.item()]))
        loss_vec = np.stack(loss_vec)
        print(e,
              np.mean(loss_vec[:, 0]),
              np.mean(loss_vec[:, 1]),
              np.mean(loss_vec[:, 2]),
              np.mean(loss_vec[:, 3]),
              np.mean(loss_vec[:, 4]),
              np.mean(loss_vec[:, 5]))
        writer.add_scalar('Loss/total', np.mean(loss_vec[:, 0]), e)
        writer.add_scalar('Loss/recon', np.mean(loss_vec[:, 1]), e)
        writer.add_scalar('Loss/step_phase', np.mean(loss_vec[:, 2]), e)
        writer.add_scalar('Loss/step_r', np.mean(loss_vec[:, 3]), e)
        writer.add_scalar('Loss/z1', np.mean(loss_vec[:, 4]), e)
        writer.add_scalar('Loss/z2', np.mean(loss_vec[:, 5]), e)
        sw = np.min([args.sw, np.mean(loss_vec[:, 2])])
        if np.mean(loss_vec[:, 2])<0.05 and np.mean(loss_vec[:, 4])<0.05: #default 0.01,0.05
            print('loss_z1を消去')
            sw1 = 0.00
            sw2 = 10.0
        else:
            sw1 = 1.0
            sw2 = 1.0
        if (e % 1) == 0:
            torch.save(enc.state_dict(),
                       f'./out/{ex_name}/enc_e{str(e).zfill(3)}.pth')
            torch.save(step.state_dict(),
                       f'./out/{ex_name}/step_e{str(e).zfill(3)}.pth')
            torch.save(dec.state_dict(),
                       f'./out/{ex_name}/dec_e{str(e).zfill(3)}.pth')
        
        if True:
            enc.eval()
            # 基準位相p0を定める
            inp = [(lc_X0[0][k]-mean_X0[k])/std_X0[k] for k in range(lc_X0.shape[1])]
            x = torch.Tensor([inp]).to(device,dtype = torch.float)
            x.requires_grad = True
            z = enc(x)
            p0 = to_polar(enc(x)[:,:2]).item()

            if lc_X0.shape[1]==2:
                #x = torch.Tensor([[1,0]]).to(device,dtype = torch.float)
                

                n = 50
                x_vec = []
                y_vec = []
                t_vec = []
                for _x in np.linspace(-2.5,2.5,n):
                    for _y in np.linspace(-2.5,2.5,n):
                        inp = [[(_x-mean_X0[0])/std_X0[0],(_y-mean_X0[1])/std_X0[1]]]
                        x = torch.Tensor(inp).to(device,dtype = torch.float)
                        x.requires_grad = True
                        z = enc(x)
                        p = to_polar(enc(x)[:,:2])
                        x_vec.append(_x)
                        y_vec.append(_y)
                        p = p.item()-p0
                        if p<0:
                            p += 2*np.pi
                        t_vec.append(p)

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                theta = np.linspace(0,2*np.pi,100)
                ax.plot(lc_X0[:, 0], lc_X0[:, 1])
                #ax.plot(np.cos(theta),np.sin(theta))
                mappable = ax.scatter(x_vec,y_vec,c=t_vec,cmap='brg')
                plt.scatter([lc_X0[0][0]], [lc_X0[0][1]], c = 'black',marker='x')
                fig.colorbar(mappable, ax=ax)
                fig.savefig(f'./out/{ex_name}/pf_e{str(e).zfill(3)}.png')
                plt.close()

            zs = []
            grad = []
            ps = []
            
            thetas = np.linspace(0, 2*np.pi, lc_X0.shape[0])
            for i in range(lc_X0.shape[0]):
                inp = [(lc_X0[i][k]-mean_X0[k])/std_X0[k] for k in range(lc_X0.shape[1])]
                x = torch.Tensor([inp]).to(device, dtype=torch.float)
                x.requires_grad = True
                z = enc(x)
                p = to_polar(enc(x)[:, :2])
                p.backward()
                ps.append(p.item())
                zs.append(z.detach().to('cpu').numpy()[0])
                grad.append(x.grad.detach().to('cpu').numpy()[0])
            zs = np.stack(zs)
            grad = np.stack(grad)
            if lc_name in LOW_DIM_LIMIT_CYCLE:
                for i in range(lc_X0.shape[1]):
                    if step.state_dict()['theta'].item() > 0:
                        plt.plot(thetas, grad[:, i]/std_X0[i], label=f'x{i+1}')
                    else:
                        plt.plot(thetas, -grad[:, i]/std_X0[i], label=f'x{i+1}')
                ymin, ymax = -1.2, 1.2
                plt.vlines(0, ymin, ymax, colors='red', linestyle='dashed')
                plt.vlines(np.pi*0.5, ymin, ymax, colors='red', linestyle='dashed')
                plt.vlines(np.pi*1.0, ymin, ymax, colors='red', linestyle='dashed')
                plt.vlines(np.pi*1.5, ymin, ymax, colors='red', linestyle='dashed')
                plt.vlines(np.pi*2.0, ymin, ymax, colors='red', linestyle='dashed')
                plt.legend()
                plt.savefig(f'./out/{ex_name}/prf_e{str(e).zfill(3)}.png')
                plt.close()

            
            # 個々の位相応答関数
            if lc_name in LOW_DIM_LIMIT_CYCLE:
                prf_score = []
                for i in range(lc_X0.shape[1]):
                    if step.state_dict()['theta'].item() > 0:
                        g = grad[:, i]/std_X0[i]
                    else:
                        g = -grad[:, i]/std_X0[i]
                    plt.plot(thetas, g, label=f'pred_x{i+1}')
                    plt.plot(thetas, lc_prf[:,i], label=f'truth_x{i+1}')
                    prf_score.append(str(np.mean(np.abs(g-lc_prf[:,i]))))
                    plt.legend()
                    plt.savefig(f'./out/{ex_name}/prf_x{i+1}_e{str(e).zfill(3)}.png')
                    plt.close()
                f = open(log_file, 'a')
                score_str = f'epoch{e} ' + ' '.join(prf_score) 
                print(score_str)
                f.write(score_str+'\n')
                f.close()

            # embeddingの確認
            ps2 = []
            for _p in ps:
                p = _p-p0
                if p<0:
                    p += 2*np.pi
                if step.state_dict()['theta'].item()<0 and p!=0:
                    p = 2*np.pi-p
                ps2.append(p)
            plt.plot([0, 2*np.pi],[0, 2*np.pi],label='truth')
            plt.scatter(thetas, ps2, s=2, color='red', label='pred')
            plt.legend()
            plt.savefig(f'./out/{ex_name}/lc_phase_e{str(e).zfill(3)}.png')
            plt.close()
    torch.save(enc.state_dict(), f'./out/{ex_name}/enc.pth')
    torch.save(step.state_dict(), f'./out/{ex_name}/step.pth')
    torch.save(dec.state_dict(), f'./out/{ex_name}/dec.pth')


if __name__ == '__main__':
    main()