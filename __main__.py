import sys
from algorithms.W_Metapath2Vec import W_Metapath2Vec


def main():
    input_file = './dataset/dblp/dblp_small.gexf'
    output_emb_file = './output/dblp/dblp_small.emb'
    # metapath = 'Author-Paper-Author'
    metapath = 'Author-Paper-Venue-Paper-Author'
    model = W_Metapath2Vec(input_file, output_emb_file, metapath)
    model.train()


if __name__ == '__main__':
    sys.exit(main())
