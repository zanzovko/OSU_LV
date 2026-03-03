file = open("song.txt", "r")

song = file.read()
song = song.split()

song_dict = {}

for i in range(len(song)):
    if song[i] in song_dict:
        song_dict[song[i]] += 1
    else:
        song_dict[song[i]] = 1

counter = 0
for i in song_dict:
    if song_dict[i] == 1:
        counter += 1

print(f"Broj različitih riječi koje se pojavljuju samo jednom: {counter}")
for i in song_dict:
    if song_dict[i] == 1:
        print(i, song_dict[i])

file.close()
