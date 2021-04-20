
TRAIN_DATA_FILE = 'persona.txt'
TEST_DATA_FILE = 'persona_test.txt'

POSITIVE_FILE = 'real.data'
TEST_FILE = 'test.data'

# Basic Training Paramters

BATCH_SIZE = 64
TOTAL_BATCH = 200
GENERATED_NUM = 10000
NEGATIVE_FILE = 'gene.data'
EVAL_FILE = 'eval.data'
PRE_EPOCH_NUM = 120

# Genrator Parameters
g_emb_dim = 500
g_hidden_dim = 500

# Discriminator Parameters
d_emb_dim = 500
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
d_dropout = 0.75
d_num_class = 2