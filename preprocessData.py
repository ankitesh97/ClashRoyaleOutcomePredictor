
batch_size = 10
data = pd.read_csv('data/small.csv').set_index('Unnamed: 0')
stats = pd.read_csv('data/cards_stats.csv').set_index('Unnamed: 0')
popu = pd.read_csv('data/popularity.csv').set_index('A')
# print stats.head()

np.random.seed(1)

data = data.sample(frac=1).reset_index(drop=True)
p1_c = ['cr1', 'cr2', 'cr3', 'cr4', 'cr5', 'cr6', 'cr7', 'cr8']
p2_c = ['cl1', 'cl2', 'cl3', 'cl4', 'cl5', 'cl6', 'cl7', 'cl8']

p1_l = ['lr1', 'lr2', 'lr3', 'lr4', 'lr5', 'lr6', 'lr7', 'lr8']
p2_l = ['ll1', 'll2', 'll3', 'll4', 'll5', 'll6', 'll7', 'll8']


def getInFormat(p_card, p_level, stats):
    
    cards_info = []
    for i in range(p_card.shape[0]):
        p_cards_info = [[],[],[],[],[],[],[],[]]
        stats_cards = stats.loc[p_card[i]]
        popularity = popu.loc[p_card[i]]
        for j in range(8):
            p_cards_info[j].append(stats_cards.loc[p_card[i][j]]['Elixer'])
            p_cards_info[j].append(float(popu.loc[p_card[i][j]]))
            p_cards_info[j].append(p_level[i][j])
            p_cards_info[j].append(stats_cards.loc[p_card[i][j]][p_level[i][j]])
        cards_info.append(p_cards_info)

    return np.array(cards_info)

def makeY(res):
    y = []
    for i in range(len(res)):
        t = [0,0,0]
        if res[i] > 0:
           t[0] = 1
        elif res[i]<0:
            t[1] = 1
        else:
            t[2] = 1
        y.append(t)
    return np.array(y)

def getBatch(data):
    data = data.sample(frac=1).reset_index(drop=True)
    batch = data[:batch_size]
    p1_cards_data = np.array(batch[p1_c])
    p1_level_data = np.array(batch[p1_l])
    p1_info =  getInFormat(p1_cards_data, p1_level_data, stats)
    p2_cards_data = np.array(batch[p2_c])
    p2_level_data = np.array(batch[p2_l])
    p2_info =  getInFormat(p2_cards_data, p2_level_data, stats)
    tl = np.array(batch['tl'])
    tr = np.array(batch['tr'])
    y = makeY(np.array(batch['result']))
    return p1_info, p2_info, tr, tl, y
