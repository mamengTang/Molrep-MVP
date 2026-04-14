#coding=utf-8
import os
import torch
import torch_geometric
from torch_geometric.nn import global_mean_pool
torch.autograd.set_detect_anomaly(True)
from torch import nn
from torch.utils.data import Dataset
import torch.optim as optim
from pcqm4m import PCQM4Mv2Dataset
from model.transformer_model import transformer_1d,AttentionPoolingWithMask
import numpy as np
from model.gnn_model import GNN,GNNDecoder
from transformers import RobertaConfig, RobertaForMaskedLM
from model.dimenet import DimeNet
from model.feature_fussion import TransformerEncoder
import torch.multiprocessing
from tqdm import tqdm
from utils import to_dense_with_fixed_padding
from utils import mask_tokens_batch2,mask_graph_batch2,add_noise_to_3d_structure_batch2
from torch_geometric.data import Batch
from torch_geometric.utils import scatter
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
BATCH_SIZE=256 
EPOCH = 20
LOAD_FROM_LAST=False
from torch.nn.modules.loss import _Loss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12362'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl",  
        init_method="env://", 
        world_size=world_size,
        rank=rank
    )

def cleanup():
    dist.destroy_process_group()


def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename, weights_only=True)

   
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['best_valid_loss']

    print(f"Checkpoint loaded from epoch {epoch}, loss: {loss}")

    return epoch, loss

class PreprocessBatch:
    def process(self, batch):

        pos = batch.pos                          
        batch_idx = batch.batch                    

        pos_mean = global_mean_pool(pos, batch_idx)  
        batch.pos = pos - pos_mean[batch_idx]       
        return batch


class ClipInfoCELoss(_Loss):
    def __init__(self, temperature=0.07):
        super(ClipInfoCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits_per_image, logits_per_text):
        sim_i2t = F.cosine_similarity(logits_per_image.unsqueeze(1), logits_per_text.unsqueeze(0), dim=2)  # (batch_size, batch_size)
        sim_t2i = F.cosine_similarity(logits_per_text.unsqueeze(1), logits_per_image.unsqueeze(0), dim=2)  # (batch_size, batch_size)
   
        sim_i2t /= self.temperature
        sim_t2i /= self.temperature
        
        labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)
        
    
        loss_i = F.cross_entropy(sim_i2t, labels) 
        loss_t = F.cross_entropy(sim_t2i, labels) 
        

        loss = (loss_i + loss_t) / 2
        return loss, labels

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder_1d = transformer_1d()
      
        self.config = RobertaConfig.from_pretrained('./roberta-base')
        self.config.hidden_size = 128
        self.config.mask_token_id = 2586
        self.config.type_vocab_size=1
        self.config.vocab_size=2586+1
        self.config.max_position_embeddings=60
        self.config.num_attention_heads = 8
        self.decoder_1d = RobertaForMaskedLM(self.config)
        self.encoder_2d = GNN(num_layer=3, hidden_dim=128,output_dim=128)
        self.decoder_2d = GNNDecoder(hidden_dim=128, out_dim=9)
        self.decoder_3d = GNNDecoder(hidden_dim=128, out_dim=3)
        self.encoder_3d = DimeNet(hidden_channels=128,
                                  num_blocks=3,
                                  num_bilinear=8,
                                  num_spherical=7,
                                  num_radial=6,
                                  out_channels=128
                                  )
        self.feature_fussion = TransformerEncoder(128, 128, 8, 4)

        self.token_bias = nn.Parameter(torch.randn(50, 128))
        self.graph_bias = nn.Parameter(torch.randn(50, 128))
        self.molecule_bias = nn.Parameter(torch.randn(50, 128))

        self.preprocessor = PreprocessBatch()
        self.fc_mu = nn.Linear(128, 128)
        self.fc_var = nn.Linear(128, 128)

        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.attention_pooling_1d = AttentionPoolingWithMask(128)
        self.attention_pooling_2d = AttentionPoolingWithMask(128)
        self.attention_pooling_3d = AttentionPoolingWithMask(128)
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, batch_data):
        batch_data = batch_data.cuda()
        self.preprocessor.process(batch_data)

        tokens_emb = torch.tensor(np.array(batch_data.tokens), dtype=torch.long).cuda()
        smi_mask = torch.tensor(np.array(batch_data.attention_mask), dtype=torch.bool).cuda()
        # =============
        batch_masked_tokens, batch_masked_token_indices=mask_tokens_batch2(tokens_emb,smi_mask)
        batch_masked_graphs, batch_masked_graph_indices, _=mask_graph_batch2(batch_data,batch_data.atom2substructure,batch_masked_token_indices)
        batch_masked_graphs = Batch.from_data_list(batch_masked_graphs)
        batch_noisy_positions,batch_noisy_positions2, _, _ = add_noise_to_3d_structure_batch2(batch_data.atom2substructure,batch_data.pos,batch_data.pos2,batch_data.batch, batch_masked_token_indices, batch_masked_graph_indices)
        # =============
        masked_token_representation_1d = self.encoder_1d(batch_masked_tokens, smi_mask)  # (batch_size, seq_length,emd_size)
        smiles_embedding = self.attention_pooling_1d(masked_token_representation_1d, ~(~smi_mask^batch_masked_token_indices))

        masked_node_representation_2d = self.encoder_2d(batch_masked_graphs.x, batch_masked_graphs.edge_index,
                                                 batch_masked_graphs.edge_attr)  # (num_nodes_in_batch, emb_dim)
        masked_node_representation_2d, mask_2d = to_dense_with_fixed_padding(masked_node_representation_2d,batch_data.batch,50)
        GNN_embedding_2d = self.attention_pooling_2d(masked_node_representation_2d, ~(~mask_2d^batch_masked_graph_indices))

        noisy_node_representation_3d = self.encoder_3d(batch_data.x[:, 0].long(), batch_noisy_positions,
                                 batch_data.batch)
        noisy_node_representation_3d_, mask_3d = to_dense_with_fixed_padding(noisy_node_representation_3d,
                                                  batch_data.batch, 50)
        GNN_embedding_3d = self.attention_pooling_3d(noisy_node_representation_3d_, ~(~mask_3d^batch_masked_graph_indices))
        
        noisy_node_representation_3d_y = self.encoder_3d(batch_data.x[:, 0].long(), batch_noisy_positions2,
                                 batch_data.batch)
        noisy_node_representation_3d__, mask_3d = to_dense_with_fixed_padding(noisy_node_representation_3d_y,
                                                  batch_data.batch, 50)
        GNN_embedding_3d_y = self.attention_pooling_3d(noisy_node_representation_3d__, ~(~mask_3d^batch_masked_graph_indices))

    

        return smiles_embedding,GNN_embedding_2d,GNN_embedding_3d,GNN_embedding_3d_y
    

def pretrain_train(train_loader,my_model,optimizer,rank):
    train_data_len = len(train_loader)
    my_model.train()
    total_loss=0
    for step, batch_data_list in enumerate(tqdm(train_loader, desc="Training", disable=True)):

        modality_1,modality_2,modality_3,modality_3_y = my_model(batch_data_list)
        
        criterion = ClipInfoCELoss()

    
        loss12,_ = criterion(modality_1, modality_2)
        loss33,_ = criterion(modality_3, modality_3_y)
        loss23,_ = criterion(modality_2, modality_3)
        loss = loss12+loss33+loss23

        total_loss +=loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), 100)
        optimizer.step()
        optimizer.zero_grad()
        if rank == 0:
        
            print(f"TRAIN: step: {step}, "
                f"loss12: {loss12.item():.2f}, "
                f"loss23: {loss23.item():.2f}, "
                f"loss33: {loss33.item():.2f}, "
                f"grad: {grad_norm.item():.2f}")    

    train_loss = total_loss/train_data_len

    return train_loss


def pretrain_evaluate(valid_loader,my_model,optimizer,rank):
    valid_data_len = len(valid_loader)
    my_model.eval()
    total_loss = 0

    total_loss12 = 0
    total_loss33 = 0
    total_loss23 = 0

    with torch.no_grad():
        for step, batch_data_list in enumerate(tqdm(valid_loader, desc="Validing", disable=True)):
            try:
                modality_1,modality_2,modality_3,modality_3_y = my_model(batch_data_list)
            except Exception as e:
                print(f"Exception at step {step}: {e}")
            
            criterion = ClipInfoCELoss()

            # 计算 InfoNCE 损失
            loss12,_ = criterion(modality_1, modality_2)
            loss33,_ = criterion(modality_3_y, modality_3)
            loss23,_ = criterion(modality_2, modality_3)

            losses = loss12+loss33+loss23

            total_loss += losses.detach().cpu()
            
            total_loss12 += loss12.detach().cpu()
            total_loss33 += loss33.detach().cpu()
            total_loss23 += loss23.detach().cpu()


        valid_loss =total_loss/valid_data_len

        mean_loss12 = total_loss12 / valid_data_len
        mean_loss33 = total_loss33 / valid_data_len
        mean_loss23 = total_loss23 / valid_data_len

        if rank == 0:
            print(f"Valid: step: {step}, "
                  f"loss12: {mean_loss12.item():.2f}, "
                  f"loss23: {mean_loss23.item():.2f}, "
                  f"loss33: {mean_loss33.item():.2f} ")  
        return valid_loss

def train_mp(rank,world_size):
    dataset = PCQM4Mv2Dataset()
    print('dataset load finish')

    randperm = torch.randperm(len(dataset))
    train_idxs = randperm[: int((0.98) * len(dataset))]
    valid_idxs = randperm[int(0.98 * len(dataset)):]


    train_loader = torch_geometric.loader.DataLoader(
        dataset[train_idxs], batch_size=BATCH_SIZE, drop_last=True, shuffle=True
    )
    valid_loader = torch_geometric.loader.DataLoader(
        dataset[valid_idxs], batch_size=BATCH_SIZE, drop_last=True, shuffle=True
    )

    train_data_len = len(train_loader)
    val_data_len = len(valid_loader)
    print('train dataset length: ', train_data_len)
    print('val dataset length: ', val_data_len)


    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    my_model = MyModel().to(rank)
    my_model = DDP(my_model, device_ids=[rank], find_unused_parameters=True)

    model_param_group = []
    model_param_group.append({'params': my_model.parameters(), 'lr': 0.0001 * 1})

    optimizer = optim.Adam(model_param_group, weight_decay=1e-5)

    best_valid_loss = 10000
    current_epoch = 0


    for epoch in range(current_epoch, EPOCH + 1):
        print('Epoch: {}'.format(epoch))
        print("Training")
        train_loss = pretrain_train(train_loader,my_model,optimizer,rank)
        print('Epoch {}, train loss: {:.4f}'.format(epoch + 1, train_loss))
        if rank == 0:
            valid_loss = pretrain_evaluate(valid_loader,my_model,optimizer,rank)
            print('Epoch {}, valid loss: {:.4f}'.format(epoch + 1, valid_loss))
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
    dist.destroy_process_group()

def main():
    world_size = 1
    mp.spawn(train_mp,
        args=(world_size),
        nprocs=world_size,
        join=True)

if __name__ == "__main__":
    main()
    