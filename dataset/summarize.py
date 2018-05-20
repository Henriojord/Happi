import random

#Read leafsnap-dataset-images.txt
with open('leafsnap-dataset/leafsnap-dataset-images.txt', 'r') as f:
    content = f.read().split('\n')[1:-1]
content = [c.split('\t') for c in content]

#Get species
species = set([c[3] for c in content])

# #Get images by species
images = {s:[] for s in species}
for c in content:
    images[c[3]].append(c[1])

#Split train / test set
dataset = {s:[[], []] for s in species}
for s in species:
    random.shuffle(images[s])
    ratio = (80 * len(images[s])) // 100
    dataset[s][0] += images[s][:ratio]
    dataset[s][1] += images[s][ratio:]

#Write summary file
for s in species:
    with open('dataset/trainset', 'a') as f:
        for t in dataset[s][0]:
            f.write('{}\t{}\n'.format(t, s))
    with open('dataset/testset', 'a') as f:
        for t in dataset[s][1]:
            f.write('{}\t{}\n'.format(t, s))
