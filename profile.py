import argparse
import time

import torch

from fast_transformers.builders import AttentionBuilder
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import FullMask, LengthMask
from fast_transformers.attention import AttentionLayer
import torch.autograd.profiler as profiler


class Timer(object):
    def __init__(self):
        if torch.cuda.is_available():
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._start = time.time()

    def measure(self):
        if torch.cuda.is_available():
            self._end.record()
            torch.cuda.synchronize()
            return self._start.elapsed_time(self._end)/1000
        else:
            return time.time()-self._start


def run(attn, x, grad):
    N = x.shape[0]
    L = x.shape[1]
    attn_mask = FullMask(L, device=x.device)
    length_mask = LengthMask(x.new_full((N,), L, dtype=torch.int64))

    # Run self attention and add it to the input

    if grad:
        y = attn(
            x, x, x,
            attn_mask=attn_mask,
            query_lengths=length_mask,
            key_lengths=length_mask
        )
        y.sum().backward()
    else:
        with torch.no_grad():
            y = attn(
                x, x, x,
                attn_mask=attn_mask,
                query_lengths=length_mask,
                key_lengths=length_mask
            )
def get_model_name(args):
    if args.attention_type == 'full':
        model = 'full'
    elif args.attention_type == 'linear':
        model = 'linear'
    elif args.attention_type == 'clustered':
        model = 'clustered-{}'.format(args.clusters)
    elif args.attention_type == 'improved-clustered':
        model = 'i-clustered-{}-{}'.format(
            args.clusters, args.topk)
    elif args.attention_type == 'improved-clustered-firstk':
        model = 'i-clustered-fk-{}-{}'.format(
            args.clusters, args.topk)
    elif args.attention_type == 'improved-clustered-pooled':
        model = 'i-clustered-pooled-{}-{}'.format(
            args.clusters, args.topk)
    elif args.attention_type == 'smyrf':
        model = 'smyrf-{}-{}'.format(
            args.clusters, args.rounds)
    elif args.attention_type == 'reformer':
        model = 'lsh-{}'.format(
            args.rounds)
    else:
        model = ''
    return model


def main(argv):
    parser = argparse.ArgumentParser(
        description="Micro benchmark the attention implementations"
    )

    parser.add_argument(
        "--nlhe",
        type=lambda x: tuple([int(xi) for xi in x.split(",")]),
        default=(8, 128, 8, 64),
        help="Set the input size to the multihead attention"
    )
    parser.add_argument(
        "--attention_type",
        default="full"
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=63,
        help="How many planes to use for hashing"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=32,
        help="How many topk keys to extract for exact dot product"
    )
    parser.add_argument(
        "--hash_bias",
        action="store_true",
        help="Use bias for the hashing"
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=100,
        help="How many clusters to use for attention acceleration"
    )
    parser.add_argument(
        "--lloyd_iterations",
        type=int,
        default=10,
        help="How many iteration to do for the K-Means algorithm"
    )
    parser.add_argument(
        "--softmax_temp",
        type=float,
        default=0.125,
        help="What temperature to use for the softmax computation"
    )
    parser.add_argument(
        "--dropout_attn",
        type=float,
        default=0.1,
        help="How much dropout to use in the attention"
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=4,
        help="How many rounds of SMYRF/Reformers attention"
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=10,
        help="Run the loop that many times to warmup the compute device"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="How many iterations to run"
    )
    parser.add_argument(
        "--with_grad",
        action="store_true",
        help="Benchmark the gradient computation as well"
    )
    args = parser.parse_args(argv)

    N, L, H, E = args.nlhe
    print("Computing for", args.attention_type, "with N,L,H,E as ", N, L, H, E)

    builder = AttentionBuilder.from_kwargs(
        attention_dropout=0.1,                   # used by softmax attention
        softmax_temp=1.,                         # used by softmax attention
        hash_bias=args.hash_bias,
        topk=args.topk,
        iterations=args.lloyd_iterations,
        clusters=args.clusters,
        q_clusters=args.clusters,
        k_clusters=args.clusters,
        rounds=args.rounds
    )

    attn = builder.get(args.attention_type)
    x = torch.randn(N, L, H, E)
    if torch.cuda.is_available():
        attn = attn.cuda()
        x = x.cuda()

    if args.with_grad:
        x = x.requires_grad_(True)

    for i in range(args.warmup_iterations):
        run(attn, x, args.with_grad)

    for i in range(args.iterations):
        run(attn, x, args.with_grad)
    
if __name__ == "__main__":
    main(None)
