file = open("SMSSpamCollection.txt", "r")
SMS = file.read()
SMS = SMS.split("\n")
print(SMS[0])

hamCounter = 0
spamCounter = 0
ham = []
spam = []
for i in range(len(SMS)):
    if SMS[i].startswith("ham"):
        hamCounter += 1
        ham.append(SMS[i])
    elif SMS[i].startswith("spam"):
        spamCounter += 1
        spam.append(SMS[i])

hamWordCount = 0
for poruka in ham:
    hamWordCount += len(poruka.split())

spamWordCount = 0
for poruka in spam:
    spamWordCount += len(poruka.split())

hamAverage = hamWordCount / hamCounter
spamAverage = spamWordCount / spamCounter
print(f"Prosječan broj riječi po ham poruci: {hamAverage:.2f}")
print(f"Prosječan broj riječi po spam poruci: {spamAverage:.2f}")

counter = 0
for i in range(len(spam)):       
    if spam[i].endswith("!"):
        counter += 1

print(f"Broj spam poruka koje završavaju uskličnikom: {counter}")