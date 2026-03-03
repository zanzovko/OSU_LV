try:
    userInput = float(input("Unesite broj u intervalu od 0.0 do 1.0: "))
    if 0.0 < userInput > 1.0:
        raise Exception("Broj nije u intervalu od 0.0 do 1.0.")
except ValueError:
    print("Neispravan unos. Molimo unesite broj.")
except Exception as e:
    print(e)


if userInput >= 0.9:
    print("A")
elif userInput >= 0.8:
    print("B")
elif userInput >= 0.7:
    print("C")
elif userInput >= 0.6:
    print("D")
else:
    print("F")