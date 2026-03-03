def CalculateNumberOfDigits(numbers):
    total_digits = 0
    for number in numbers:
        total_digits += 1
    
    return total_digits
 

numbers = []
while True:
    print("Unesite Done kada želite završiti unos: ")
    userInput = input("Unesite vrijednost: ")

    if userInput == "Done":
        break

    try:
        number = float(userInput)
        numbers.append(number)
    except ValueError:
        print("Neispravan unos. Pokušajte ponovo.")


total_digits = CalculateNumberOfDigits(numbers)
print(f"Ukupan broj znamenki: {total_digits}")
average = sum(numbers) / total_digits
print(f"Prosjek: {average}")
min_value = min(numbers)
print(f"Najmanja vrijednost: {min_value}")
max_value = max(numbers)
print(f"Najveća vrijednost: {max_value}")
numbers.sort()
print(f"Sortirani brojevi: {numbers}")
