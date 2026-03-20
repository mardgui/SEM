import argparse
import json

import clip
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


def generate_random_list(n, m, seed):
    torch.manual_seed(seed)
    return torch.randint(0, m , (n,)).tolist()


def generate_variable_random_list(classes, class_indices, class_descriptions, seed):
    torch.manual_seed(seed)
    random_list = []
    for class_idx in class_indices:
        n_desc = len(class_descriptions[classes[class_idx]])
        random_list.append(torch.randint(0, n_desc, (1,)).item())
    return random_list


def generate_complementary_random_list(num_samples, classes, class_descriptions, seed):
    torch.manual_seed(seed)
    random_list = []
    for _ in range(num_samples):
        sub_list = []
        for c in classes:
            n_desc = len(class_descriptions[c])
            sub_list.append(torch.randint(0, n_desc, (1,)).item())
        random_list.append(sub_list)
    return random_list


def replace_a_with_an(text):
    words = text.split()
    for i in range(len(words) - 1):
        if words[i].lower() == 'a' and words[i + 1][0].lower() in 'aeiou':
            words[i] = 'an'
    return ' '.join(words)


class PRISMDataset(Dataset):
    def __init__(self, scene_descriptions,
                 target_classes, bias_classes,
                 tc_descriptions, bc_descriptions,
                 target_placeholder, bias_placeholder,
                 num_samples, seed):
        super().__init__()
        self.sd = scene_descriptions
        self.tc = target_classes
        self.bc = bias_classes
        self.tcd = tc_descriptions
        self.bcd = bc_descriptions
        self.ns = num_samples
        
        self.sd_idx = generate_random_list(self.ns, len(self.sd), seed)
        self.tc_idx = generate_random_list(self.ns, len(self.tc), seed)
        self.bc_idx = generate_random_list(self.ns, len(self.bc), seed)
        self.tcd_idx = generate_variable_random_list(self.tc, self.tc_idx, self.tcd, seed)
        self.tcd_complementary_bcd_idx = generate_complementary_random_list(self.ns, self.bc, self.bcd, seed)
        self.bcd_idx = generate_variable_random_list(self.bc, self.bc_idx, self.bcd, seed)
        self.bcd_complementary_tcd_idx = generate_complementary_random_list(self.ns, self.tc, self.tcd, seed)

        self.tp = target_placeholder
        self.bp = bias_placeholder

    def __len__(self):
        return self.ns
    
    def __getitem__(self, index):
        scene_description = self.sd[self.sd_idx[index]]
        target_class = self.tc[self.tc_idx[index]]
        bias_class = self.bc[self.bc_idx[index]]
        target_class_description = self.tcd[target_class][self.tcd_idx[index]]
        bias_class_description = self.bcd[bias_class][self.bcd_idx[index]]

        positives = []  # for positives: same target class, different bias class -> maximize sim
        for i, bc in enumerate(self.bc):
            bcd = self.bcd[bc][self.tcd_complementary_bcd_idx[index][i]]
            positives.append(replace_a_with_an(scene_description.replace(self.tp, target_class_description).replace(self.bp, bcd)))

        negatives = []  # for negatives: same bias class, different target class -> minimize sim
        for i, tc in enumerate(self.tc):
            tcd = self.tcd[tc][self.bcd_complementary_tcd_idx[index][i]]
            negatives.append(replace_a_with_an(scene_description.replace(self.tp, tcd).replace(self.bp, bias_class_description)))

        return positives, negatives 


def get_SSL_dataset(args):
    try:
        with open(f'debiasing_methods/prism_data/{args.task}/{args.setting}.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f'Unknown or unsupported task: {args.task}')
        exit(1)

    textset = PRISMDataset(config['scene descriptions'],
                           config['target classes'],
                           config['bias classes'],
                           config['target class descriptions'],
                           config['bias class descriptions'],
                           config['target placeholder'],
                           config['bias placeholder'],
                           args.num_samples,
                           args.seed)
    textloader = DataLoader(textset, batch_size=args.batch_size, shuffle=True)
    
    return textset, textloader


class EmbeddingTransformer(torch.nn.Module):
    def __init__(self, embed_dim=768, identity=False):
        super(EmbeddingTransformer, self).__init__()
        self.transformer = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        if identity:
            self.transformer.weight.data = torch.eye(embed_dim, dtype=torch.float32)

    def forward(self, x):
        x = self.transformer(x)
        return x


class EmbeddingOrthogonalizer(torch.nn.Module):
    def __init__(self, embed_dim=768, num_bases=10):
        super(EmbeddingOrthogonalizer, self).__init__()
        self.bases = torch.nn.Parameter(torch.randn(num_bases, embed_dim))

    def forward(self, x):
        V = self.bases.T
        V = V.to(torch.float32)
        VtV = V.T @ V
        VtV_inv = torch.inverse(VtV)
        P = torch.eye(V.shape[0], device=V.device, dtype=torch.float32) - V @ VtV_inv @ V.T
        return x @ P


def get_transformer(CLIP_model, num_bases, init_weight, device):  
    if CLIP_model.startswith('RN50'):
        embed_dim = 1024
    elif CLIP_model.startswith('ViT-L/14'):
        embed_dim = 768
    elif CLIP_model in ['ViT-B/16', 'RN101']:
        embed_dim = 512

    if num_bases == 0:
        if init_weight == 'i':
            transformer = EmbeddingTransformer(embed_dim=embed_dim, identity=True)
        else:
            transformer = EmbeddingTransformer(embed_dim=embed_dim)
    else:
        transformer = EmbeddingOrthogonalizer(embed_dim=embed_dim, num_bases=num_bases)
        
    transformer.to(device)
    return transformer


def orth_transformation_calculation(args, model, spurious_words):
    spurious_tokens = clip.tokenize(spurious_words).to(args.device)
    with torch.no_grad():
        spurious_embeddings = model.encode_text(spurious_tokens)
        spurious_embeddings /= spurious_embeddings.norm(dim=-1, keepdim=True)

    V = spurious_embeddings.T
    V = V.to(torch.float32)
    VtV = V.T @ V
    VtV_inv = torch.inverse(VtV)
    P = torch.eye(V.shape[0], device=args.device, dtype=torch.float32) - V @ VtV_inv @ V.T
    return P


def sentence_list_to_embedding(args, model, sentences):
    tokens = torch.concat([clip.tokenize(sentence).to(args.device) for sentence in sentences], dim=0)
    return F.normalize(model.encode_text(tokens), dim=-1)


def contrastive_loss(pos_embeddings, neg_embeddings, margin=0.6):
    pos_embeddings = F.normalize(pos_embeddings, dim=-1)
    mean_pos_embedding = F.normalize(torch.mean(pos_embeddings, dim=1, keepdim=True), dim=-1)
    pos_sim = F.cosine_similarity(pos_embeddings, mean_pos_embedding, dim=-1)
    pos_loss = (1 - pos_sim).mean()

    B, N, _ = neg_embeddings.shape
    neg_embeddings = F.normalize(neg_embeddings, dim=-1)

    neg_losses = []
    neg_indices = torch.triu_indices(N, N, offset=1)
    for i in range(B):
        i_neg_embeddings = neg_embeddings[i]
        neg_sim_matrix = torch.matmul(i_neg_embeddings, i_neg_embeddings.T)
        neg_pairwise_sims = neg_sim_matrix[neg_indices[0], neg_indices[1]]
        neg_loss = torch.relu(neg_pairwise_sims - margin).mean()
        neg_losses.append(neg_loss)

    return pos_loss + torch.stack(neg_losses).mean()


def train_transformation(args, model, textloader, transformer):
    optimizer = torch.optim.Adam(transformer.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(args.epochs):
        total_loss = 0
        iter = 0
        for positives, negatives in tqdm(textloader):
            optimizer.zero_grad()

            with torch.no_grad():
                pos_embeddings = torch.stack([sentence_list_to_embedding(args, model, sentence) for sentence in positives], dim=1).detach()
                neg_embeddings = torch.stack([sentence_list_to_embedding(args, model, sentence) for sentence in negatives], dim=1).detach()

            transformed_pos_embeddings = F.normalize(transformer(pos_embeddings.float()), dim=-1)
            transformed_neg_embeddings = F.normalize(transformer(neg_embeddings.float()), dim=-1)
            
            loss = contrastive_loss(transformed_pos_embeddings, transformed_neg_embeddings)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
            iter += 1

        print(f'Epoch: {epoch}, Loss: {total_loss / iter}')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--device', type=str.lower, default='cuda')
    args.add_argument('--CLIP_model', type=str, default='ViT-L/14@336px',
                      help='CLIP model to use [ViT-B/16, ViT-L/14@336px, ...]')
    args.add_argument('--task', type=str.lower, default='waterbirds',
                      help='task [retrieval, zs_classification]')
    args.add_argument('--setting', type=str.lower, default='celeba_blond_hair_gender',
                      help='task setting to use [celeba_blond_hair_gender, waterbirds, ...]')
    args.add_argument('--epochs', type=int, default=1,
                      help='number of epochs to train the embedding transformer')
    args.add_argument('--num_bases', type=int, default=0,
                      help='Free transformation if zero otherwise number of bases for orthogonalization')
    args.add_argument('--num_samples', type=int, default=500,
                      help='Number of samples to use for training the transformation')
    args.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    args.add_argument('--lr', type=float, default=1e-1,
                      help='Learning rate for training the transformation')
    args.add_argument('--batch_size', type=int, default=64,
                      help='Batch size during training')
    args.add_argument('--wd', type=float, default=0.0,
                      help='Weight decay for training the transformation')
    args.add_argument('--init_weight', type=str.lower, default='random',
                      help='How to initialize the weights [random, orth, I]')

    args = args.parse_args()

    saved_model_file = f'prism_clip={args.CLIP_model.replace("/", "-")}_task={args.task}_setting={args.setting}'
    saved_model_file += f'_epochs={args.epochs}_num_bases={args.num_bases}_num_samples={args.num_samples}_seed={args.seed}'
    saved_model_file += f'_lr={args.lr}_batch_size={args.batch_size}_wd={args.wd}_init_weight={args.init_weight}.pth'
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, preprocess = clip.load(args.CLIP_model, device=args.device)

    _, text_loader = get_SSL_dataset(args)
    transformer = get_transformer(args.CLIP_model, args.num_bases, args.init_weight, args.device)
    if args.init_weight == 'orth' and args.num_bases == 0:
        if args.task == 'celeba_blond_hair_gender':
            spurious_words = ['Man', 'Woman']
        elif args.task == 'waterbirds':
            spurious_words = ['water', 'land']
        else:
            print(f'Unknown or unsupported task: {args.task}')
            exit(1)
        P = orth_transformation_calculation(args, model, spurious_words)
        transformer.transformer.weight.data = P

    train_transformation(args, model, text_loader, transformer)
    torch.save(transformer.state_dict(), f'debiasing_methods/prism_weights/{saved_model_file}')
